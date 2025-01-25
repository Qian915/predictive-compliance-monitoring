# Predictive Compliance Monitoring

## Description
Code accompanying the paper Quantifying The Degree Of Process Compliance: Predictive Compliance Monitoring Approaches

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
python3 data_processing.py
```

Predict compliance states and quantify the degree of compliance
```
python3 pcm.py
```

## License
LGPL-3.0 license
