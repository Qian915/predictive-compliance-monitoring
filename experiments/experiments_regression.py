import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import time
import os
import sys
from sys import argv
import pickle
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


dataset_ref = argv[1]
params_dir = argv[2]
results_dir = argv[3]
bucket_method = argv[4]
cls_encoding = argv[5]
cls_method = argv[6]
gap = int(argv[7])
n_iter = int(argv[8])

if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"

method_name = "%s_%s"%(bucket_method, cls_encoding)

dataset_ref_to_datasets = {
    "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2"],
    "o2c": ["o2c"],
    "bpic2012w": ["bpic2012w_1", "bpic2012w_2"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]
    
train_ratio = 0.8
random_state = 22

# create params and results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))

if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))
    
for dataset_name in datasets:
    print(f'\nfor {dataset_name}:')

    # load optimal params
    optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
        continue
        
    with open(optimal_params_filename, "rb") as fin:
        args = pickle.load(fin)
        print(f'hyperparams:\n{args}')
    
    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    
    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True}

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
        print(f'max_prefix_length: {max_prefix_length}')
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
        print(f'max_prefix_length: {max_prefix_length}')

    # split into training and test
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    if gap > 1:
        outfile = os.path.join(results_dir, "%s_%s_%s_gap%s_acc.csv" % (cls_method, dataset_name, method_name, gap))
    else:
        outfile = os.path.join(results_dir, "%s_%s_%s_acc.csv" % (cls_method, dataset_name, method_name))
        
    start_test_prefix_generation = time.time()
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
    test_prefix_generation_time = time.time() - start_test_prefix_generation
            
    offline_total_times = []
    online_event_times = []
    train_prefix_generation_times = []
    for ii in range(n_iter):
        # create prefix logs
        start_train_prefix_generation = time.time()
        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length, gap)

        train_prefix_generation_time = time.time() - start_train_prefix_generation
        train_prefix_generation_times.append(train_prefix_generation_time)
            
        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method':bucket_encoding, 
                         'case_id_col':dataset_manager.case_id_col, 
                         'cat_cols':[dataset_manager.activity_col], 
                         'num_cols':[], 
                         'random_state':random_state}
        if bucket_method == "cluster":
            bucketer_args["n_clusters"] = int(args["n_clusters"])
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)

        start_offline_time_bucket = time.time()
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)  # train the bucketer and assign a bucket to each case: [bucket1, bucket3,...]
        offline_time_bucket = time.time() - start_offline_time_bucket

        bucket_assignments_test = bucketer.predict(dt_test_prefixes)    # assign a bucket to each case in the test set based on the trained bucketer

        pred_y_class = []
        pred_y_reg = []
        test_y_class = []
        test_y_values = []
        nr_events_all = []
        offline_time_fit = 0
        current_online_event_times = []
        #print(f'iteration {ii}: #buckets in test set: {len(set(bucket_assignments_test))}')
        for bucket in set(bucket_assignments_test):
            if bucket_method == "prefix":
                current_args = args[bucket]
            else:
                current_args = args
            relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
            relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket) # a subset of prefixes where all cases are within the same bucket
            
            nr_events_all.extend(list(dataset_manager.get_prefix_lengths(dt_test_bucket)))
            if len(relevant_train_cases_bucket) == 0:
                preds = [dataset_manager.get_class_ratio(train)] * len(relevant_test_cases_bucket)
                current_online_event_times.extend([0] * len(preds))
            else:
                dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_train_cases_bucket) # one row per event 
                train_y_values = dataset_manager.get_regression_label(dt_train_bucket)   #TODO for regression
                # scale train_y within [0, 1]
                y_scaler = MinMaxScaler()
                train_y_reg = y_scaler.fit_transform(train_y_values.to_numpy().reshape(-1, 1)).ravel()  # normalization for regression task
            
                if len(set(train_y_reg)) < 2:
                    preds = [train_y_reg[0]] * len(relevant_test_cases_bucket)
                    pred_y_reg.extend(preds)
                    current_online_event_times.extend([0] * len(preds))
                    test_y_values.extend(dataset_manager.get_regression_label(dt_test_bucket)) #TODO for regression
                else:
                    start_offline_time_fit = time.time()    # [(static, static_encoder, state, state_encoder)] for case and dynamic attributes
                    feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])

                    if cls_method == "rf":
                        cls = RandomForestClassifier(n_estimators=500,
                                                     max_features=current_args['max_features'],
                                                     random_state=random_state)

                    elif cls_method == "xgboost":
                        cls = xgb.XGBRegressor(objective='reg:squarederror',    #TODO for regression
                                                n_estimators=500,
                                                learning_rate= current_args['learning_rate'],
                                                subsample=current_args['subsample'],
                                                max_depth=int(current_args['max_depth']),
                                                colsample_bytree=current_args['colsample_bytree'],
                                                min_child_weight=int(current_args['min_child_weight']),
                                                seed=random_state)

                    elif cls_method == "logit":
                        cls = LogisticRegression(C=2**current_args['C'],
                                                 random_state=random_state)

                    elif cls_method == "svm":
                        cls = SVC(C=2**current_args['C'],
                                  gamma=2**current_args['gamma'],
                                  random_state=random_state)

                    if cls_method == "svm" or cls_method == "logit":
                        pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                    else:
                        pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])

                    train_x = dt_train_bucket.drop(columns=["label", "magnitude"])  #TODO prepare input features for model training
                    pipeline.fit(train_x, train_y_reg)
                    offline_time_fit += time.time() - start_offline_time_fit

                    # predict separately for each prefix case
                    test_all_grouped = dt_test_bucket.groupby(dataset_manager.case_id_col)
                    for _, group in test_all_grouped:
                        test_y_values.extend(dataset_manager.get_regression_label(group))
                            
                        start = time.time()
                        _ = bucketer.predict(group)
                        test_x = group.drop(columns=["label", "magnitude"])  #TODO prepare input features for prediction
                        if cls_method == "svm":
                            pred = pipeline.decision_function(test_x)
                            pred_y_reg.extend(pred)
                        else:
                            pred = pipeline.predict(test_x)
                            pred_y_reg.extend(pred)

                        pipeline_pred_time = time.time() - start
                        current_online_event_times.append(pipeline_pred_time / len(group))
                        
        pred_y_reg = np.clip(pred_y_reg, 0, 1)      #TODO clip predicted normalized values within [0,1]           
        pred_y_values = y_scaler.inverse_transform(np.array(pred_y_reg).reshape(-1, 1)).ravel()   # TODO inverse transformation: return original value
        offline_total_time = offline_time_bucket + offline_time_fit + train_prefix_generation_time
        offline_total_times.append(offline_total_time)
        online_event_times.append(current_online_event_times)

        
    with open(outfile, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s;%s\n"%("dataset", "method", "cls", "nr_events", "n_iter", "metric", "score"))

        fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, -1, "test_prefix_generation_time", test_prefix_generation_time))

        for ii in range(len(offline_total_times)):
            fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, ii, "train_prefix_generation_time", train_prefix_generation_times[ii]))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, ii, "offline_time_total", offline_total_times[ii]))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, ii, "online_time_avg", np.mean(online_event_times[ii])))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, ii, "online_time_std", np.std(online_event_times[ii])))
        
        test_y_class = [1 if value > 0 else 0 for value in test_y_values]
        pred_y_class = [1 if value > 0 else 0 for value in pred_y_values]
        dt_results = pd.DataFrame({"actual_values": test_y_values, "actual": test_y_class, "predicted": pred_y_class, "pred_values": pred_y_values, "nr_events": nr_events_all})
        dt_results.to_csv(f'truth_prediction/{dataset_name}_regression.csv', index=False)

        for nr_events, group in dt_results.groupby("nr_events"):
            if len(set(group.actual)) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, -1, "auc", np.nan))
                fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, -1, "mae", np.nan))
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "baseline_mae_mean", np.nan))
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "baseline_mae_median", np.nan))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, -1, "auc", roc_auc_score(group.actual, group.predicted)))   
                fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events, -1, "mae", mean_absolute_error(group.actual_values, group.pred_values)/1440))
                baseline_mae_mean = mean_absolute_error(group.actual_values, [np.mean(group.actual_values)]*len(group.actual_values))/1440  # mae in days
                baseline_mae_median = mean_absolute_error(group.actual_values, [np.median(group.actual_values)]*len(group.actual_values))/1440  # mae in days
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "baseline_mae_mean", baseline_mae_mean))
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "baseline_mae_median", baseline_mae_median))

        overall_auc = roc_auc_score(dt_results.actual, dt_results.predicted)
        overall_f1 = f1_score(dt_results.actual, dt_results.predicted)
        overall_mae = mean_absolute_error(dt_results.actual_values, dt_results.pred_values)/1440  # mae in days
        baseline_mae_mean = mean_absolute_error(dt_results.actual_values, [np.mean(dt_results.actual_values)]*len(dt_results.actual_values))/1440  # mae in days
        baseline_mae_median = mean_absolute_error(dt_results.actual_values, [np.median(dt_results.actual_values)]*len(dt_results.actual_values))/1440  # mae in days

        print(f'AUC: {overall_auc:.2f}')
        print(f'F1: {overall_f1:.2f}')
        print(f'Model MAE: {overall_mae:.2f}')
        print(f'Baseline MAE mean: {baseline_mae_mean:.2f}')
        print(f'Baseline MAE median: {baseline_mae_median:.2f}')

        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "overall_auc", overall_auc))
        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "overall_f1", overall_f1))
        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "overall_mae", overall_mae))
        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "baseline_mae_mean", baseline_mae_mean))
        fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "baseline_mae_median", baseline_mae_median))

        online_event_times_flat = [t for iter_online_event_times in online_event_times for t in iter_online_event_times]
        fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, -1, "online_time_avg", np.mean(online_event_times_flat)))
        fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, -1, "online_time_std", np.std(online_event_times_flat)))
        fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, -1, "offline_time_total_avg", np.mean(offline_total_times)))
        fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, -1, -1, "offline_time_total_std", np.std(offline_total_times)))