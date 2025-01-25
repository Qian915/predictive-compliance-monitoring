import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.metrics import AUC
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense, Attention, GlobalAveragePooling1D   #TODO use Att-Bi-LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import time
import os
import pickle
from sys import argv


# Objective function to minimize
def objective(params):
    print("Training with parameters:", params)
    
    # 3-Fold Cross-Validation Setup
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    #auc_scores = []
    mae_scores = []
    val_losses = []
    
    # Split the data into 3 parts
    for train_index, val_index in kfold.split(train_x):
        # Split data for the current fold
        x_train_fold, x_val_fold = train_x[train_index], train_x[val_index]
        y_train_values_scaled_fold, y_val_values_scaled_fold = train_y_values_scaled[train_index], train_y_values_scaled[val_index]
        
        # Build the Att-Bi-LSTM model
        def create_regression_model(input_shape, lstm_units, dropout_rate, dense_dropout):
            input_layer = Input(shape=input_shape)

            # BiLSTM layer
            bi_lstm_out = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(input_layer)
            bi_lstm_out = Dropout(rate=dropout_rate)(bi_lstm_out)

            # Attention Layer
            attention_out = Attention()([bi_lstm_out, bi_lstm_out])  # Self-attention (query, key, value = bi_lstm_out)
            attention_out = GlobalAveragePooling1D()(attention_out)  # Pooling the attention output to reduce the sequence to a single vector (for binary classification)
            dense_out = Dense(64, activation='relu')(attention_out)
            dense_out = Dropout(rate=dense_dropout)(dense_out)

            # Output for regression task (linear for continuous values)
            regression_output = Dense(1, activation='relu', name='regression_output')(dense_out)      #TODO linear -> relu: constraint to non-negative regression output!

            # Define the model with the two outputs
            model = Model(inputs=input_layer, outputs=regression_output)

            return model
        '''
        def create_multitask_model(input_shape, lstm_units=params['lstm_units'], dropout_rate=params['dropout_rate'], dense_dropout=params['dense_dropout']):
            input_layer = Input(shape=input_shape)
            lstm_out = LSTM(units=lstm_units, return_sequences=False)(input_layer)
            lstm_out = Dropout(rate=dropout_rate)(lstm_out)
            dense_out = Dense(64, activation='relu')(lstm_out)
            dense_out = Dropout(rate=dense_dropout)(dense_out)
            # Output for regression task (linear for continuous values)
            regression_output = Dense(1, activation='linear')(dense_out)
            # Define the model with the two outputs
            model = Model(inputs=input_layer, outputs=regression_output)
            
            return model
        '''
        input_shape = (max_prefix_length, feature_dim)
        model = create_regression_model(input_shape, params['lstm_units'], params['dropout_rate'], params['dense_dropout']) #TODO define hyper params. for optimization here
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)       #TODO define early stoppping with patience of 5 to avoid over-fitting
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']), 
                    loss='mean_squared_error', 
                    metrics=['mae'])

        # Train the model with the current fold
        history = model.fit(
            x_train_fold, y_train_values_scaled_fold,   # for classification and regression!
            validation_data=(x_val_fold, y_val_values_scaled_fold),
            epochs=epochs,
            batch_size=params['batch_size'],
            verbose=0,
            callbacks=[early_stopping]      #TODO add early stopping
        )

        # Get validation MAE for this fold
        val_mae = history.history['val_mae'][-1]  #TODO MAE
        mae_scores.append(val_mae)
        #val_loss = history.history['val_loss'][-1]
        #val_losses.append(val_loss)
    
    # Average AUC over the 3 folds
    #mean_auc = np.mean(auc_scores)
    #mean_loss = np.mean(val_losses)
    mean_mae = np.mean(mae_scores)
    
    # Return the negative AUC to minimize (since Hyperopt minimizes the objective)
    return {'loss': mean_mae, 'status': STATUS_OK}
    #return {'loss': mean_loss, 'status': STATUS_OK}


# Set parameters
dataset_ref = argv[1]
params_dir = argv[2]
bucket_method = argv[3]
cls_encoding = argv[4]
cls_method = argv[5]
epochs = int(argv[6])

method_name = f"{bucket_method}_{cls_encoding}"

dataset_ref_to_datasets = {
    "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2"],
    "o2c": ["o2c"],
    "bpic2012w": ["bpic2012w_1", "bpic2012w_2"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]

train_ratio = 0.8
random_state = 22

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))
    
for dataset_name in datasets:
    print(f"\nfor {dataset_name}:")

    # Read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()

    # Determine prefix lengths
    min_prefix_length = 1
    max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

    # Split into training and test
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length).reset_index(drop=True)
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length).reset_index(drop=True)
    
    #train_y_class = np.array(dataset_manager.get_class_label(dt_train_prefixes))  # k-prefixes
    train_y_values = np.array(dataset_manager.get_regression_label(dt_train_prefixes))
    #test_y_class = np.array(dataset_manager.get_class_label(dt_test_prefixes))
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
        
    # Define the hyperparameter search space
    space = {
        'lstm_units': hp.choice('lstm_units', [25, 50, 75, 100, 150]),      # modified lstm units
        'dropout_rate': hp.uniform('dropout_rate', 0.01, 0.5),      # modified dropout rate
        'dense_dropout': hp.uniform('dense_dropout', 0.01, 0.5),    # modified dropout rate
        'learning_rate': hp.loguniform('learning_rate', -5, -2),    # Log scale for learning rate
        'batch_size': hp.choice('batch_size', [8, 16, 32, 64])      # batch size as a hyperparameter
    }

    # optimize parameters
    trial_nr = 1
    trials = Trials()
    fout_all = open(os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name)), "w")
    if "prefix" in method_name:
        fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "nr_events", "param", "value", "score"))   
    else:
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "param", "value", "score"))   
    best = fmin(
        fn=objective,          # The objective function to minimize
        space=space,           # The hyperparameter search space
        algo=tpe.suggest,      # Use the TPE algorithm for optimization
        max_evals=epochs,          # Maximum number of evaluations
        trials=trials          # Store the trials in the trials object
    )
    print("Best hyperparameters found:", best)
    fout_all.close()

    # write the best parameters
    best_params = hyperopt.space_eval(space, best)
    outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    # write to file
    with open(outfile, "wb") as fout:
        pickle.dump(best_params, fout)
