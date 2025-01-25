import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense, Attention, GlobalAveragePooling1D, Concatenate   #TODO use Att-Bi-LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

import time
import os
import pickle
from sys import argv

# Set parameters
dataset_ref = argv[1]
params_dir = argv[2]
results_dir = argv[3]
bucket_method = argv[4]
cls_encoding = argv[5]
gap = int(argv[6])
epochs = int(argv[7])

method_name = f"{bucket_method}_{cls_encoding}"

dataset_ref_to_datasets = {
    "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2"],
    "o2c": ["o2c"],
    "bpic2012w": ["bpic2012w_1", "bpic2012w_2"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]

train_ratio = 0.8
random_state = 22

# Create params and results directories
os.makedirs(params_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
cls_method = "lstm"

for dataset_name in datasets:
    print(f"\nfor {dataset_name}:")

    # load optimal params
    optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
        continue
        
    with open(optimal_params_filename, "rb") as fin:
        args = pickle.load(fin)
        print(f'hyperparams:\n{args}')

    # Read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()

    # Determine prefix lengths
    min_prefix_length = 1
    max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

    # Split into training and test
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length, gap).reset_index(drop=True)
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length).reset_index(drop=True)
    
    train_y_class = np.array(dataset_manager.get_class_label(dt_train_prefixes))  # k-prefixes
    train_y_values = np.array(dataset_manager.get_regression_label(dt_train_prefixes))
    test_y_class = np.array(dataset_manager.get_class_label(dt_test_prefixes))
    test_y_values = np.array(dataset_manager.get_regression_label(dt_test_prefixes))

    # Encode training and test data
    dynamic_cat_cols = dataset_manager.dynamic_cat_cols
    static_cat_cols = dataset_manager.static_cat_cols
    dynamic_num_cols = dataset_manager.dynamic_num_cols
    static_num_cols = dataset_manager.static_num_cols
    transformers = []
    if dynamic_cat_cols or static_cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), dynamic_cat_cols + static_cat_cols))
    if dynamic_num_cols or static_num_cols:
        transformers.append(('num', StandardScaler(), dynamic_num_cols + static_num_cols))
    preprocessor = ColumnTransformer(transformers)

    train_features = dt_train_prefixes[dynamic_cat_cols + static_cat_cols + dynamic_num_cols + static_num_cols]
    train_x_encoded = preprocessor.fit_transform(train_features)
    test_features = dt_test_prefixes[dynamic_cat_cols + static_cat_cols + dynamic_num_cols + static_num_cols]
    test_x_encoded = preprocessor.transform(test_features)
    
    # Reshaping the input data to (prefixes, max_len, features)
    grouped_train = dt_train_prefixes.groupby(dataset_manager.case_id_col)
    grouped_test = dt_test_prefixes.groupby(dataset_manager.case_id_col)

    def group_to_sequences(grouped_df, encoded_features, max_seq_len):
        sequences = []
        for _, group in grouped_df:
            # Align indices with encoded features
            group_indices = group.index  # Indices of the group in the original DataFrame
            group_encoded = encoded_features[group_indices.to_list()]  # Get encoded rows for this group
            
            # If sparse, convert to dense
            if hasattr(group_encoded, "todense"):
                group_encoded = group_encoded.todense()
            
            sequences.append(group_encoded)
        
        # Pad sequences to ensure uniform length
        padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post', dtype='float32')
        return padded_sequences

    
    train_x = group_to_sequences(grouped_train, train_x_encoded, max_seq_len=max_prefix_length)
    test_x = group_to_sequences(grouped_test, test_x_encoded, max_seq_len=max_prefix_length)
    feature_dim = preprocessor.transform(train_features).shape[1]   # feature dimensions afer encoding

    # Normalize numerical values
    y_scaler = MinMaxScaler()
    train_y_values_scaled = y_scaler.fit_transform(train_y_values.reshape(-1, 1))
    
    # Build the Att-Bi-LSTM model
    def create_multitask_model(input_shape, lstm_units, dropout_rate, dense_dropout):
        input_layer = Input(shape=input_shape)

        # BiLSTM layer
        bi_lstm_out = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(input_layer)
        bi_lstm_out = Dropout(rate=dropout_rate)(bi_lstm_out)

        # Attention Layer
        attention_out = Attention()([bi_lstm_out, bi_lstm_out])  # Self-attention (query, key, value = bi_lstm_out)
        attention_out = GlobalAveragePooling1D()(attention_out)  # Pooling the attention output to reduce the sequence to a single vector (for binary classification)
        dense_out = Dense(64, activation='relu')(attention_out)
        dense_out = Dropout(rate=dense_dropout)(dense_out)

        # Output for binary classification task (sigmoid)
        binary_output = Dense(1, activation='sigmoid', name='binary_output')(dense_out)

        # Output for regression task (linear for continuous values)
        regression_concat = Concatenate()([dense_out, binary_output])
        regression_output = Dense(1, activation='relu', name='regression_output')(regression_concat)      #TODO linear -> relu: constraint to non-negative regression output!

        # Define the model with the two outputs
        model = Model(inputs=input_layer, outputs=[binary_output, regression_output])

        return model
    '''
    def create_multitask_model_with_interplay(input_shape, lstm_units, dropout_rate, dense_dropout):
        input_layer = Input(shape=input_shape)

        # BiLSTM layer
        bi_lstm_out = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(input_layer)
        bi_lstm_out = Dropout(rate=dropout_rate)(bi_lstm_out)

        # Attention Layer
        attention_out = Attention()([bi_lstm_out, bi_lstm_out])
        attention_out = GlobalAveragePooling1D()(attention_out)
        shared_dense = Dense(64, activation='relu')(attention_out)
        shared_dense = Dropout(rate=dense_dropout)(shared_dense)

        # Binary Classification Task
        #binary_dense = Dense(64, activation='relu')(shared_dense)  #TODO don't separate tasks -> same architecture as the baseline
        binary_output = Dense(1, activation='sigmoid', name='binary_output')(shared_dense)

        # Regression Task: Incorporate classification output
        regression_concat = Concatenate()([shared_dense, binary_output])
        regression_dense = Dense(64, activation='relu')(regression_concat)
        #regression_output = Dense(1, activation='linear', name='regression_output')(regression_dense)
        regression_output = Dense(1, activation='relu', name='regression_output')(regression_dense)     #TODO constraint to non-negative regression output!

        # Classification Task: Incorporate regression output
        #classification_concat = Concatenate()([shared_dense, regression_output])
        #classification_dense = Dense(64, activation='relu')(classification_concat)
        #refined_binary_output = Dense(1, activation='sigmoid', name='refined_binary_output')(classification_dense)

        model = Model(inputs=input_layer, outputs=[binary_output, regression_output])
        return model
    
    # Dynamic Loss Weights
    class DynamicLossWeights(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            self.log_sigma1 = tf.Variable(0.0, trainable=True, dtype=tf.float32)  # Classification
            self.log_sigma2 = tf.Variable(0.0, trainable=True, dtype=tf.float32)  # Regression

        def call(self, loss1, loss2):
            weight1 = tf.exp(-self.log_sigma1)
            weight2 = tf.exp(-self.log_sigma2)
            return (
                weight1 * loss1 + self.log_sigma1 +
                weight2 * loss2 + self.log_sigma2
            )

    # Masked Regression Loss: Apply loss only for violations
    
    def masked_regression_loss(y_true, y_pred, binary_output):
        mask = tf.cast(tf.greater(binary_output, 0.5), tf.float32)  # Mask for violated cases
        return tf.reduce_mean(mask * tf.square(y_true - y_pred))  # Only penalize for violations
    
    def masked_regression_loss(y_true_reg, y_pred_reg, y_true_class, y_pred_class):
        # Mask for cases where classification is correct
        correct_class_mask = tf.cast(tf.equal(y_true_class, tf.round(y_pred_class)), tf.float32)
        # Mask for violation cases (binary output > 0.5)
        violation_mask = tf.cast(tf.greater(y_pred_class, 0.5), tf.float32)
        # Combine both masks
        combined_mask = correct_class_mask * violation_mask
        # Calculate masked regression loss
        return tf.reduce_mean(combined_mask * tf.square(y_true_reg - y_pred_reg))
    
    # Custom loss that uses both classification and regression tasks with dynamic loss weights
    def multitask_loss(y_true, y_pred, loss_layer):
        # Separate classification and regression targets from y_true
        y_true_class = y_true[0]  # Assuming y_true is a list: [classification_labels, regression_labels]
        y_true_reg = y_true[1]

        # Separate predictions for classification and regression
        y_pred_class = y_pred[0]
        y_pred_reg = y_pred[1]
        
        # Calculate classification loss (binary crossentropy)
        classification_loss = tf.keras.losses.binary_crossentropy(y_true_class, y_pred_class)
        
        # Calculate regression loss with masking for violation cases
        #regression_loss = masked_regression_loss(y_true_reg, y_pred_reg, y_pred_class)
        regression_loss = tf.reduce_mean(tf.square(y_true_reg - y_pred_reg))    #TODO handle outliers(e.g., many small values and few large ones): regression_loss = tf.reduce_mean(tf.square(y_true_reg - y_pred_reg) / (1 + tf.abs(y_true_reg)))

        # Combine both losses with dynamic uncertainty weighting
        return loss_layer(classification_loss, regression_loss)
    '''
    
    input_shape = (max_prefix_length, feature_dim)
    model = create_multitask_model(input_shape, lstm_units=args['lstm_units'], dropout_rate=args['dropout_rate'], dense_dropout=args['dense_dropout'])
    model.compile(optimizer=Adam(learning_rate=args['learning_rate']), 
                  loss={'binary_output': 'binary_crossentropy', 'regression_output': 'mean_squared_error'}, 
                  metrics={'binary_output': ['AUC'], 'regression_output': ['mae']})
    #model = create_multitask_model_with_interplay(input_shape, lstm_units=args['lstm_units'], dropout_rate=args['dropout_rate'], dense_dropout=args['dense_dropout'])
    #loss_layer = DynamicLossWeights()
    #model.compile(
        #optimizer=Adam(learning_rate=args['learning_rate']),
        #loss=lambda y_true, y_pred: multitask_loss(
        #    y_true, y_pred, loss_layer  # Pass y_true (both classification and regression labels) and y_pred
        #),
        #metrics={
        #    'binary_output': ['AUC'],   #TODO decide on the output layer!
        #    'regression_output': ['mae']
        #}
    #)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)       #TODO define early stoppping with patience of 5 to avoid over-fitting
    #model.summary()

    # Train the model
    start_training_time = time.time()
    history = model.fit(
        train_x, [train_y_class, train_y_values_scaled],  #{'refined_binary_output': train_y_class, 'regression_output': train_y_values_scaled},   #TODO modified for refined_binary_output
        validation_split=0.2,
        epochs=epochs,
        batch_size=args['batch_size'],
        verbose=0,
        callbacks=[early_stopping]      #TODO add early stopping
    )
    training_time = time.time() - start_training_time

    # Evaluate the model
    test_predictions = model.predict(test_x)
    pred_y_class = test_predictions[0].round().flatten()  # Binary predictions (rounded to 0 or 1)
    pred_y_values_scaled = test_predictions[1]  # Raw predictions (for regression)
    pred_y_values = y_scaler.inverse_transform(pred_y_values_scaled)

    auc = roc_auc_score(test_y_class, pred_y_class)
    f1 = f1_score(test_y_class, pred_y_class)
    mae = mean_absolute_error(test_y_values, pred_y_values)/1440
    baseline_mae = mean_absolute_error(test_y_values, np.full_like(test_y_values, np.mean(test_y_values)))/1440
    median_mae = mean_absolute_error(test_y_values, np.full_like(test_y_values, np.median(test_y_values))) / 1440

    print(f"AUC: {auc:.2f}")
    print(f"F1: {f1:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f'Baseline MAE: {baseline_mae:.2f}')
    print(f'Medium MAE: {median_mae:.2f}')

    # Save results
    results_path = os.path.join(results_dir, f"{dataset_name}_mtl.csv")
    with open(results_path, 'w') as fout:
        fout.write(f"Dataset: {dataset_name}\n")
        fout.write(f"Training Time: {training_time}\n")
        fout.write(f"AUC: {auc:.2f}\n")
        fout.write(f"F1: {f1:.2f}\n")
        fout.write(f"MAE: {mae:.2f}\n")
        fout.write(f'Baseline MAE: {baseline_mae:.2f}')
        fout.write(f'Medium MAE: {median_mae:.2f}')
