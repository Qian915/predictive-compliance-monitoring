from DatasetManager import DatasetManager
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Attention, Concatenate, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import tensorflow.keras.backend as K
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from tensorflow.keras.preprocessing.sequence import pad_sequences
import hyperopt

from sys import argv
import os
import pickle

# Define the custom masked MSE loss for the regression task
def masked_mse(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())  # Mask for non-zero values
    return K.sum(mask * K.square(y_true - y_pred)) / (K.sum(mask) + K.epsilon())

# Define the multi-task model architecture
def create_multitask_model(input_shape, lstm_units, dropout_rate, dense_dropout):
        input_layer = Input(shape=input_shape)

        # BiLSTM layer
        bi_lstm_out = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(input_layer)
        bi_lstm_out = Dropout(rate=dropout_rate)(bi_lstm_out)

        # Attention Layer
        attention_out = Attention()([bi_lstm_out, bi_lstm_out])  
        attention_out = GlobalAveragePooling1D()(attention_out)  
        dense_out = Dense(64, activation='relu')(attention_out)
        shared_features = Dropout(rate=dense_dropout)(dense_out)

        # Output for binary classification task
        binary_dense = Dense(32, activation='relu')(shared_features)
        binary_output = Dense(1, activation='sigmoid', name='binary_output')(shared_features)

        # Output for regression task
        regression_concat = Concatenate()([shared_features, binary_dense])    # input for regression task: shared features and features derived from the classification task
        regression_output = Dense(1, activation='relu', name='regression_output')(regression_concat)      # relu: constraint to non-negative regression output!

        # Define the model with the two outputs
        model = Model(inputs=input_layer, outputs=[binary_output, regression_output])

        return model

# Define the objective function for hyperparameter optimization
def objective(params):
    global trial_nr
    trial_nr += 1
    print("Training with parameters:", params)
    
    # K-Fold Cross-Validation Setup
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []
    mae_scores = []

    for train_index, val_index in kfold.split(train_x):
        # Split data for the current fold
        x_train_fold, x_val_fold = train_x[train_index], train_x[val_index]
        y_train_class_fold, y_val_class_fold = train_y_class[train_index], train_y_class[val_index]
        y_train_values_scaled_fold, y_val_values_scaled_fold = train_y_values_scaled[train_index], train_y_values_scaled[val_index]
        
        # Build the multi-task model
        input_shape = (max_prefix_length, feature_dim)
        model = create_multitask_model(
            input_shape=input_shape,
            lstm_units=params['lstm_units'],
            dropout_rate=params['dropout_rate'],
            dense_dropout=params['dense_dropout']
        )

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss={
                'binary_output': 'binary_crossentropy', 
                'regression_output': masked_mse
            },
            loss_weights={
                'binary_output': 1.0,  # Classification loss weight
                'regression_output': 0.5  # Regression loss weight
            },
            metrics={
                'binary_output': ['AUC'], 
                'regression_output': ['mae']
            }
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        history = model.fit(
            x_train_fold, 
            {
                'binary_output': y_train_class_fold, 
                'regression_output': y_train_values_scaled_fold
            },
            validation_data=(
                x_val_fold, 
                {
                    'binary_output': y_val_class_fold, 
                    'regression_output': y_val_values_scaled_fold
                }
            ),
            epochs=50,
            batch_size=params['batch_size'],
            verbose=0,
            callbacks=[early_stopping],
        )

        # Extract the validation metrics
        val_auc = history.history.get('val_binary_output_auc', [0])[-1]  # AUC value from validation
        auc_scores.append(val_auc)
        val_mae = history.history.get('val_regression_output_mae', [0])[-1]  # MAE value from validation
        mae_scores.append(val_mae)

    # Calculate mean AUC and MAE across folds
    mean_auc = np.mean(auc_scores)
    mean_mae = np.mean(mae_scores)

    # Normalize AUC and MAE for combined loss
    normalized_auc = mean_auc  
    normalized_mae = mean_mae  
    alpha = 0.5  # Weight for AUC
    beta = 0.5  # Weight for MAE
    total_loss = -alpha * normalized_auc + beta * normalized_mae

    # keep track of trials
    fout_all.write(f"{trial_nr};{dataset_name};{cls_method};{method_name};{params};{total_loss:.4f}\n")

    return {'loss': total_loss, 'status': STATUS_OK}


# Set parameters
dataset_ref = argv[1]
params_dir = argv[2]
n_iter = int(argv[3])
bucket_method = argv[4]
cls_encoding = argv[5]
cls_method = argv[6]

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
    train_y_class = np.array(dataset_manager.get_class_label(dt_train_prefixes))  # k-prefixes
    train_y_values = np.array(dataset_manager.get_regression_label(dt_train_prefixes))
    
    # Encode training data
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
    
    # Reshaping the input data to (prefixes, max_len, features)
    grouped_train = dt_train_prefixes.groupby(dataset_manager.case_id_col)
    def group_to_sequences(grouped_df, encoded_features, max_seq_len):
        sequences = []
        for _, group in grouped_df:
            group_indices = group.index  
            group_encoded = encoded_features[group_indices.to_list()]
            if hasattr(group_encoded, "todense"):
                group_encoded = group_encoded.todense()
            sequences.append(group_encoded)
        padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post', dtype='float32')
        return padded_sequences
    train_x = group_to_sequences(grouped_train, train_x_encoded, max_seq_len=max_prefix_length)
    feature_dim = preprocessor.transform(train_features).shape[1]   # feature dimensions afer encoding
    y_scaler = StandardScaler()
    train_y_values_scaled = y_scaler.fit_transform(train_y_values.reshape(-1, 1))   # Normalize numerical values

    ### hyperparameter optimization ###
    space = {
        'lstm_units': hp.choice('lstm_units', [32, 64, 128]),
        'dropout_rate': hp.uniform('dropout_rate', 0.01, 0.5),
        'dense_dropout': hp.uniform('dense_dropout', 0.01, 0.5),
        'learning_rate': hp.loguniform('learning_rate', -5, -2),
        'batch_size': hp.choice('batch_size', [16, 32, 64])
    }

    # Initialize trials and log output for hyperparameter optimization
    trial_nr = 1
    trials = Trials()

    fout_all = open(os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name)), "w")
    fout_all.write("iter;dataset;cls;method;param;value;score\n")

    # Run the hyperparameter optimization
    best = fmin(
        fn=objective,  
        space=space,  
        algo=tpe.suggest,  # Use the TPE algorithm for optimization
        max_evals=n_iter,  
        trials=trials  
    )
    fout_all.close()

    # Write the best parameters to a file
    best_params = hyperopt.space_eval(space, best)
    outfile = os.path.join(params_dir, f"optimal_params_{cls_method}_{dataset_name}_{method_name}.pickle")
    with open(outfile, "wb") as fout:
        pickle.dump(best_params, fout)

    # Print the best parameters
    print("Best hyperparameters:", best_params)
