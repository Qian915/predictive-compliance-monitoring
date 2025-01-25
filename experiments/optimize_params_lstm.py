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
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import time
import os
import pickle
from sys import argv


# Objective function to minimize
def objective(params):
    print("Training with parameters:", params)
    
    # Build the model using the received hyperparameters
    model = Sequential([
        LSTM(units=params['lstm_units'], input_shape=(1, train_x.shape[-1]), return_sequences=False),
        Dropout(rate=params['dropout_rate']),
        Dense(64, activation='relu'),
        Dropout(rate=params['dense_dropout']),
        Dense(1, activation='sigmoid')  # For binary classification
    ])

    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model (using validation split or a validation dataset)
    history = model.fit(
        train_x, train_y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=0
    )

    # Get the validation loss (or other metric like validation accuracy)
    val_loss = history.history['val_loss'][-1]

    # Return the loss to minimize
    return {'loss': val_loss, 'status': STATUS_OK}


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
    #"o2c": ["o2c"],
    #"bpic2012w": ["bpic2012w_1", "bpic2012w_2"]
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
        
    # Define the hyperparameter search space
    space = {
        'lstm_units': hp.choice('lstm_units', [32, 64, 128]),
        'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.5),
        'dense_dropout': hp.uniform('dense_dropout', 0.2, 0.5),
        'learning_rate': hp.loguniform('learning_rate', -5, -2),  # Log scale for learning rate
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
