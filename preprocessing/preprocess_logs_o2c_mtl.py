import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from statistics import median, mean

input_data_folder = "data/orig_logs"
output_data_folder = "data/labeled_logs_csv_processed"
in_filename = "o2c.csv"

case_id_col = "case:concept:name"
activity_col = "concept:name"
timestamp_col = "time:timestamp"
label_col = "label"
magnitude_col = "magnitude"
pos_label = "deviant"
neg_label = "regular"

category_freq_threshold = 10

# features for classifier
dynamic_cat_cols = ["concept:name"] # i.e. event attributes
static_cat_cols = [] # i.e. case attributes that are known from the start
dynamic_num_cols = []
static_num_cols = []

static_cols = static_cat_cols + static_num_cols + [case_id_col]
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col]
cat_cols = dynamic_cat_cols + static_cat_cols


def extract_timestamp_features(group):
    
    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')
    
    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    tmp = tmp.fillna(pd.Timedelta(0))  # Ensure timedelta type
    group["timesincelastevent"] = tmp.apply(lambda x: round(float(x / np.timedelta64(1, 'm')), 2)) # m is for minutes

    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    tmp = tmp.fillna(pd.Timedelta(0))  # Ensure timedelta type
    group["timesincecasestart"] = tmp.apply(lambda x: round(float(x / np.timedelta64(1, 'm')), 2)) # m is for minutes with 2 decimals

    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    group["event_nr"] = range(1, len(group) + 1)
    
    return group

def get_open_cases(date):
    return sum((dt_first_last_timestamps["start_time"] <= date) & (dt_first_last_timestamps["end_time"] > date))
    
def check_if_both_exist_and_time_less_than(group, pre, suc, time_limit):  # group == trace
    pre_idxs = np.where(group[activity_col] == pre)[0]
    suc_idxs = np.where(group[activity_col] == suc)[0]
    # both occur
    if len(pre_idxs) > 0 and len(suc_idxs) > 0:
        pre_idx = pre_idxs[0]   # assume only one occurrence for both activities
        suc_idx = suc_idxs[0]
        # check time
        if pre_idx < suc_idx:
            time_actual = group["timesincecasestart"].iloc[suc_idx] - group["timesincecasestart"].iloc[pre_idx]
            if time_actual <= time_limit:
                # time satisfied
                group[label_col] = neg_label
                group[magnitude_col] = 0
            else:
                # time violated: calculate magnitude of violation = time_actual - time_limit
                group[label_col] = pos_label
                group[magnitude_col] = round(time_actual - time_limit, 2)
            return group[:suc_idx]  # cut trace before suc occurs
        else:
            #TODO activity violation: suc before pre -> magnitude of violation = total time ???
            group[label_col] = pos_label
            group[magnitude_col] = round(group["timesincecasestart"].iloc[-1], 2)
            return group
    # pre is absent: vacuously satisfied
    elif len(pre_idxs) == 0:
        group[label_col] = neg_label
        group[magnitude_col] = 0
        return group
    #TODO pre occures but not followed by suc: activity violation -> magnitude of violation = total time ???
    else:
        group[label_col] = pos_label
        group[magnitude_col] = round(group["timesincecasestart"].iloc[-1], 2)
        return group

def check_if_any_of_activities_exist(group, activities):
    if np.sum(group[activity_col].isin(activities)) > 0:
        return True
    else:
        return False
    
data = pd.read_csv(os.path.join(input_data_folder, in_filename))
data[case_id_col] = data[case_id_col].fillna("missing_caseid")

# remove incomplete cases
tmp = data.groupby(case_id_col).apply(check_if_any_of_activities_exist, activities=["archive order", "reject order"])   # end activities
incomplete_cases = tmp.index[tmp==False]
data = data[~data[case_id_col].isin(incomplete_cases)]

data = data[static_cols + dynamic_cols]

# add features extracted from timestamp
data[timestamp_col] = pd.to_datetime(data[timestamp_col])
data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
data["month"] = data[timestamp_col].dt.month
data["weekday"] = data[timestamp_col].dt.weekday
data["hour"] = data[timestamp_col].dt.hour
data = data.groupby(case_id_col).apply(extract_timestamp_features)
# Reset index to make sure 'Case ID' becomes a regular column
data = data.reset_index(drop=True)

# add inter-case features
data = data.sort_values([timestamp_col], ascending=True, kind='mergesort')
dt_first_last_timestamps = data.groupby(case_id_col)[timestamp_col].agg([min, max])
dt_first_last_timestamps.columns = ["start_time", "end_time"]
data["open_cases"] = data[timestamp_col].apply(get_open_cases)

# impute missing values
grouped = data.sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(case_id_col)
for col in static_cols + dynamic_cols:
    data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))
        
data[cat_cols] = data[cat_cols].fillna('missing')
data = data.fillna(0)
    
# set infrequent factor levels to "other"
for col in cat_cols:
    counts = data[col].value_counts()
    mask = data[col].isin(counts[counts >= category_freq_threshold].index)
    data.loc[~mask, col] = "other"

# first labeling
dt_time_calculated = data.sort_values(timestamp_col, ascending=True, kind="mergesort").groupby(case_id_col).apply(check_if_both_exist_and_time_less_than, pre="confirm order", suc="ship goods", time_limit=1440)    # max: 24h -> 1440 minutes
dt_time_calculated = dt_time_calculated.reset_index(drop=True)
dt_time_calculated.to_csv(os.path.join(output_data_folder, "o2c.csv"), sep=",", index=False)