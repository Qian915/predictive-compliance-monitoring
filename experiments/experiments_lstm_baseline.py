import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense, Attention, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

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
    train_y = np.array(dataset_manager.get_class_label(dt_train_prefixes))  # k-prefixes
    test_y = np.array(dataset_manager.get_class_label(dt_test_prefixes))

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
            group_indices = group.index  
            group_encoded = encoded_features[group_indices.to_list()]  
            if hasattr(group_encoded, "todense"):
                group_encoded = group_encoded.todense()
            sequences.append(group_encoded)
        # Pad sequences to ensure uniform length
        padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post', dtype='float32')
        return padded_sequences
    
    train_x = group_to_sequences(grouped_train, train_x_encoded, max_seq_len=max_prefix_length)
    test_x = group_to_sequences(grouped_test, test_x_encoded, max_seq_len=max_prefix_length)
    feature_dim = preprocessor.transform(train_features).shape[1]   # feature dimensions afer encoding

    # Build the Att-Bi-LSTM model
    def create_baseline_model(input_shape, lstm_units, dropout_rate, dense_dropout):
        input_layer = Input(shape=input_shape)

        # BiLSTM layer
        bi_lstm_out = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(input_layer)
        bi_lstm_out = Dropout(rate=dropout_rate)(bi_lstm_out)

        # Attention Layer
        attention_out = Attention()([bi_lstm_out, bi_lstm_out])  
        attention_out = GlobalAveragePooling1D()(attention_out) 
        dense_out = Dense(64, activation='relu')(attention_out)
        dense_out = Dropout(rate=dense_dropout)(dense_out)

        # Output for binary classification task (sigmoid)
        binary_output = Dense(1, activation='sigmoid', name='binary_output')(dense_out)

        # Define the model with the two outputs
        model = Model(inputs=input_layer, outputs=binary_output)
        return model
    
    input_shape = (max_prefix_length, feature_dim)
    model = create_baseline_model(input_shape, lstm_units=args['lstm_units'], dropout_rate=args['dropout_rate'], dense_dropout=args['dense_dropout'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)       # define early stoppping with patience of 10 to avoid over-fitting
    model.compile(optimizer=Adam(learning_rate=args['learning_rate']), 
                  loss='binary_crossentropy', 
                  metrics=['AUC'])

    # Train the model
    start_training_time = time.time()
    history = model.fit(
        train_x, train_y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=args['batch_size'],  
        verbose=0,
        callbacks=[early_stopping]      
    )
    training_time = time.time() - start_training_time

    # Evaluate the model
    test_predictions = model.predict(test_x).round().flatten()

    auc = roc_auc_score(test_y, test_predictions)
    print(f"AUC: {auc:.2f}")