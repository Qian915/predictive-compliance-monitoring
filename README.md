# Predictive Compliance Monitoring

## Description
Code accompanying the paper Quantifying The Magnitude Of Violation: Predictive Compliance Monitoring Approaches

## Getting started

Create a virtual environment
```
python3 -m venv venv
```

Activate your virtual environment
```
source venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

Process training and test data sets
```
python3 preprocessing/preprocess_logs_sepsis_mtl.py
```

Predict compliance states and quantify the magnitude of violation for baseline/hybrid/MTL approaches with xgboost
```
python3 experiments/experiments_baseline.py "sepsis_cases" "params_baseline/" "results_baseline/" "single" "agg" "xgboost" 1 20
python3 experiments/experiments_hybrid.py "sepsis_cases" "params_hybrid/" "results_hybrid/" "single" "agg" "xgboost" 1 20
python3 experiments/experiments_mtl.py "sepsis_cases" "params_mtl/" "results_mtl/" "single" "agg" "xgboost" 1 20
```
Predict compliance states and quantify the magnitude of violation for baseline/hybrid/MTL approaches with Att-Bi-LSTM
```
python3 experiments/experiments_lstm_baseline.py "sepsis_cases" "params_baseline/lstm/" "results_lstm/" "single" "index" 1 50
python3 experiments/experiments_lstm_hybrid.py "sepsis_cases" "params_hybrid/lstm/" "results_lstm/" "single" "index" 1 50
python3 experiments/experiments_lstm_mtl.py "sepsis_cases" "params_mtl/lstm/" "results_lstm/" "single" "index" 1 50
```

## Citation
This project incorporates and builds upon the work:
Teinemaa, I., Dumas, M., La Rosa, M., & Maggi, F. M. (2019). Outcome-Oriented Predictive Process Monitoring: Review and Benchmark. TKDD, 13(2), 17:1–17:57.

## License
LGPL-3.0 license
