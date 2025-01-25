import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, mean_absolute_error
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

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample


def create_and_evaluate_model(args):
    global trial_nr
    trial_nr += 1
    
    start = time.time()
    score = 0
    mae = 0
    total_loss = 0
    for cv_iter in range(n_splits):
        
        dt_test_prefixes = dt_prefixes[cv_iter]
        dt_train_prefixes = pd.DataFrame()
        for cv_train_iter in range(n_splits): 
            if cv_train_iter != cv_iter:
                dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)
        
        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method':bucket_encoding, 
                         'case_id_col':dataset_manager.case_id_col, 
                         'cat_cols':[dataset_manager.activity_col], 
                         'num_cols':[], 
                         'random_state':random_state}
        if bucket_method == "cluster":
            bucketer_args["n_clusters"] = args["n_clusters"]
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
        bucket_assignments_test = bucketer.predict(dt_test_prefixes)
        
        pred_y_class = []
        pred_y_reg = []
        test_y_class = []
        test_y_values = []
        if "prefix" in method_name:
            scores = defaultdict(int)
        for bucket in set(bucket_assignments_test):
            relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
            relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)
            test_y_class = dataset_manager.get_class_label(dt_test_bucket)    #TODO for classification
            test_y_values.extend(dataset_manager.get_regression_label(dt_test_bucket))   #TODO for regression
            if len(relevant_train_cases_bucket) == 0:
                preds = [class_ratios[cv_iter]] * len(relevant_test_cases_bucket)
            else:
                dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_train_cases_bucket) # one row per event
                train_y_class = dataset_manager.get_class_label(dt_train_bucket)   #TODO for classification 
                train_y_values = dataset_manager.get_regression_label(dt_train_bucket)   #TODO for regression
                # scale train_y within [0, 1]
                y_scaler = MinMaxScaler()
                train_y_reg = y_scaler.fit_transform(train_y_values.to_numpy().reshape(-1, 1)).ravel()  # normalization for regression task
                # Combine labels into a single dataset for MTL
                train_y_mtl = np.column_stack((train_y_class, train_y_reg))     #TODO combine outputs for multi-task learning
                
                if len(set(train_y_class)) < 2:
                    preds = [train_y_class[0]] * len(relevant_test_cases_bucket)
                    pred_y_class.extend(preds)
                    pred_y_reg.extend(preds)
                else:
                    feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])

                    if cls_method == "rf":
                        cls = RandomForestClassifier(n_estimators=500,
                                                     max_features=args['max_features'],
                                                     random_state=random_state)

                    elif cls_method == "xgboost":
                        def custom_mtl_loss(preds, dtrain):
                            """
                            Custom loss function combining classification (logloss) and regression (MSE).
                            """
                            labels = dtrain.get_label().reshape(-1, 2)
                            preds_class = preds[:, 0]  # Classification predictions
                            preds_reg = preds[:, 1]    # Regression predictions
                            
                            labels_class = labels[:, 0]  # True labels for classification
                            labels_reg = labels[:, 1]    # True labels for regression

                            # Classification loss (binary log-loss)
                            class_loss = -labels_class * np.log(preds_class) - (1 - labels_class) * np.log(1 - preds_class)

                            # Regression loss (mean squared error)
                            reg_loss = (preds_reg - labels_reg) ** 2

                            # Weighted sum of the two losses
                            combined_loss = np.mean(class_loss) + np.mean(reg_loss)
                            return 'mtl_loss', combined_loss
                        
                        cls = xgb.XGBRegressor(objective='reg:logistic',  #TODO Use logistic regression for classification
                                                eval_metric=custom_mtl_loss,  #TODO Custom loss function for MTL
                                                n_estimators=500,
                                                learning_rate= args['learning_rate'],
                                                subsample=args['subsample'],
                                                max_depth=int(args['max_depth']),
                                                colsample_bytree=args['colsample_bytree'],
                                                min_child_weight=int(args['min_child_weight']),
                                                seed=random_state)

                    elif cls_method == "logit":
                        cls = LogisticRegression(C=2**args['C'],
                                                 random_state=random_state)

                    elif cls_method == "svm":
                        cls = SVC(C=2**args['C'],
                                  gamma=2**args['gamma'],
                                  random_state=random_state)

                    if cls_method == "svm" or cls_method == "logit":
                        pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                    else:
                        pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                    #pipeline.fit(dt_train_bucket, train_y)
                    train_x = dt_train_bucket.drop(columns=["label", "magnitude"])  #TODO prepare input features for model training
                    pipeline.fit(train_x, train_y_mtl)
                    test_x = dt_test_bucket.drop(columns=["label", "magnitude"])  #TODO prepare input features for prediction

                    if cls_method == "svm":
                        preds = pipeline.decision_function(test_x)
                    else:
                        preds = pipeline.predict(test_x)
                        pred_class, pred_reg = preds[:, 0], preds[:, 1]  # Split predictions
                        pred_y_class.extend(pred_class)
                        pred_y_reg.extend(pred_reg)

            pred_y_class = np.array(pred_y_class)
            pred_y_class = (pred_y_class > 0.5).astype(int)  # convert the regression output to binary class predictions
            pred_y_class = pred_y_class.tolist()            
            pred_y_reg = np.clip(pred_y_reg, 0, 1)      #TODO clip predicted normalized values within [0,1]           
            pred_y_values = y_scaler.inverse_transform(np.array(pred_y_reg).reshape(-1, 1)).ravel()   # TODO inverse transformation: return original value
            if "prefix" in method_name:
                auc = 0.5
                if len(set(test_y_values)) == 2: 
                    auc = roc_auc_score(test_y_class, pred_y_class)
                scores[bucket] += auc
        #score += roc_auc_score(test_y_class, pred_y_class)
                
        # Metrics calculation
        mae += mean_absolute_error(test_y_values, pred_y_values)/1440  # mae in days
        auc = roc_auc_score(test_y_class, pred_y_class)
        # Combine metrics
        alpha, beta = 0.5, 0.5  # Adjust based on importance
        total_loss += alpha * (1 - auc) + beta * mae
    
    if "prefix" in method_name:
        for k, v in args.items():
            for bucket, bucket_score in scores.items():
                fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, bucket, k, v, bucket_score / n_splits))   
        fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, 0, "processing_time", time.time() - start, 0))  
    else:
        for k, v in args.items():
            fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, k, v, total_loss / n_splits))      #TODO score (score / n_splits) -> total loss
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, "processing_time", time.time() - start, 0))   
    fout_all.flush()
    #return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}
    return {'loss': total_loss / n_splits, 'status': STATUS_OK, 'model': cls}


dataset_ref = argv[1]
params_dir = argv[2]
n_iter = int(argv[3])
bucket_method = argv[4]
cls_encoding = argv[5]
cls_method = argv[6]

if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"

method_name = "%s_%s"%(bucket_method, cls_encoding)

dataset_ref_to_datasets = {
    #"sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2"],
    #"o2c": ["o2c"],
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
n_splits = 3
random_state = 22

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))
    
for dataset_name in datasets:
    
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
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

    # split into training and test
    train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    
    # prepare chunks for CV
    dt_prefixes = []
    class_ratios = []
    for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
        class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
        # generate data where each prefix is a separate instance
        dt_prefixes.append(dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length))
    del train
        
    # set up search space
    if cls_method == "rf":
        space = {'max_features': hp.uniform('max_features', 0, 1)}
    elif cls_method == "xgboost":
        space = {'learning_rate': hp.uniform("learning_rate", 0, 1),
                 'subsample': hp.uniform("subsample", 0.5, 1),
                 'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                 'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                 'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}
    elif cls_method == "logit":
        space = {'C': hp.uniform('C', -15, 15)}
    elif cls_method == "svm":
        space = {'C': hp.uniform('C', -15, 15),
                 'gamma': hp.uniform('gamma', -15, 15)}
    if bucket_method == "cluster":
        space['n_clusters'] = scope.int(hp.quniform('n_clusters', 2, 50, 1))

    # optimize parameters
    trial_nr = 1
    trials = Trials()
    fout_all = open(os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name)), "w")
    if "prefix" in method_name:
        fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "nr_events", "param", "value", "loss"))
    else:
        fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "param", "value", "loss"))      #TODO score -> loss
    best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)
    fout_all.close()

    # write the best parameters
    best_params = hyperopt.space_eval(space, best)
    outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    # write to file
    with open(outfile, "wb") as fout:
        pickle.dump(best_params, fout)
