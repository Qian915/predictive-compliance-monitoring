import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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
    #"o2c": ["o2c"],
    #"bpic2012w": ["bpic2012w_1", "bpic2012w_2"]
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
    data[dataset_manager.activity_col] = data[dataset_manager.activity_col].str.lower()
    data[dataset_manager.activity_col] = data[dataset_manager.activity_col].str.replace(" ", "-")
    activities = list(data[dataset_manager.activity_col].unique())
    val = range(len(activities))
    x_word_dict = dict(zip(activities, val))

    # Determine prefix lengths
    min_prefix_length = 1
    max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

    # Split into training and test
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

    # Prepare train and test prefixes
    def generate_prefixes(df, dataset_manager, split):
        case_id, activities = dataset_manager.case_id_col, dataset_manager.activity_col
        processed_df = pd.DataFrame(columns = ["Case ID", "prefix", "k", "timesincelastevent", "timesincecasestart", "label", "magnitude"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][activities].to_list()
            last_time = df[df[case_id] == case]["timesincelastevent"].to_list()
            case_time = df[df[case_id] == case]["timesincecasestart"].to_list()
            label = df[df[case_id] == case]["label"].to_list()[0]
            magnitude = df[df[case_id] == case]["magnitude"].to_list()[0]
            for i in range(len(act) - 1):
                #prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                prefix = act[0] if i == 0 else " ".join(act[:i+1])
                processed_df.at[idx, "Case ID"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "timesincelastevent"] = last_time[i]
                processed_df.at[idx, "timesincecasestart"] = case_time[i]
                processed_df.at[idx, "label"] = 1 if label == "deviant" else 0
                processed_df.at[idx, "magnitude"] = magnitude
                idx = idx + 1
        processed_df.to_csv(f'data/labeled_prefixes/{dataset_name}_{split}.csv', index=False)
        return processed_df
    train_prefixes = generate_prefixes(train, dataset_manager, "train")
    test_prefixes = generate_prefixes(test, dataset_manager, "test")

    # tokenization
    def prepare_data_compliance(df, x_word_dict, max_case_length, time_scaler = None, y_scaler = None, shuffle = True):

        x = df["prefix"].values
        time_x = df[["timesincelastevent", "timesincecasestart"]].values.astype(np.float32)
        y = df["label"].values

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        if time_scaler is None:
            time_scaler = StandardScaler()
            time_x = time_scaler.fit_transform(
                time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(
                time_x).astype(np.float32)            

        token_x = pad_sequences(token_x, maxlen=max_case_length)
        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return token_x, time_x, y, time_scaler
    
    # prepare training data
    train_x_encoded, train_x_numeric, train_y, time_scaler = prepare_data_compliance(train_prefixes, x_word_dict, max_prefix_length)
    train_x = np.concatenate([train_x_encoded, train_x_numeric], axis=1)

    # prepare test data
    test_x_encoded, test_x_numeric, test_y, _ = prepare_data_compliance(test_prefixes, x_word_dict, max_prefix_length, time_scaler)
    test_x = np.concatenate([test_x_encoded, test_x_numeric], axis=1)

    # Reshaping the input data to 3D (batch_size, timesteps, features)
    train_x = np.expand_dims(train_x, axis=1)  # Shape becomes (batch_size, 1, features)
    test_x = np.expand_dims(test_x, axis=1)    # Shape becomes (batch_size, 1, features)

    print(f"Shape of train_x: {train_x.shape}")
    print(f"Shape of test_x: {test_x.shape}")
    print(f"Shape of train_y: {train_y.shape}")
    print(f"Shape of test_y: {test_y.shape}")

    # Build the LSTM model
    model = Sequential([
        # Input layer: Use None for the sequence length (dynamic input shape)
        LSTM(units=args['lstm_units'], input_shape=(1, train_x.shape[-1]), return_sequences=False),
        Dropout(rate=args['dropout_rate']),
        Dense(64, activation='relu'),
        Dropout(rate=args['dense_dropout']),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=args['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    start_training_time = time.time()
    history = model.fit(
        train_x, train_y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    training_time = time.time() - start_training_time

    # Evaluate the model
    test_predictions = model.predict(test_x).round().flatten()
    auc = roc_auc_score(test_y, test_predictions)
    f1 = f1_score(test_y, test_predictions)

    print(f"AUC: {auc}")
    print(f"F1: {f1}")

    # Save results
    results_path = os.path.join(results_dir, f"{dataset_name}_baseline.csv")
    with open(results_path, 'w') as fout:
        fout.write(f"Dataset: {dataset_name}\n")
        fout.write(f"Training Time: {training_time}\n")
        fout.write(f"AUC: {auc}\n")
        fout.write(f"F1: {f1}\n")
