import os

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
magnitude_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

logs_dir = "data/labeled_logs_csv_processed"

#### Sepsis Cases settings ####
datasets = ["sepsis_cases_%s" % i for i in range(1, 3)]

for dataset in datasets:
    
    filename[dataset] = os.path.join(logs_dir, "%s.csv" % dataset)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "org:group"
    timestamp_col[dataset] = "time:timestamp"
    label_col[dataset] = "label"
    magnitude_col[dataset] = "magnitude"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", 'org:group'] # i.e. event attributes
    static_cat_cols[dataset] = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                       'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                       'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                       'SIRSCritTemperature', 'SIRSCriteria2OrMore'] # i.e. case attributes that are known from the start
    dynamic_num_cols[dataset] = ['CRP', 'LacticAcid', 'Leucocytes', "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['Age']

#### O2C Cases settings ####
dataset = "o2c"

filename[dataset] = os.path.join(logs_dir, "o2c.csv")

case_id_col[dataset] = "case:concept:name"
activity_col[dataset] = "concept:name"
timestamp_col[dataset] = "time:timestamp"
label_col[dataset] = "label"
magnitude_col[dataset] = "magnitude"
pos_label[dataset] = "deviant"
neg_label[dataset] = "regular"

# features for classifier
static_cat_cols[dataset] = []
static_num_cols[dataset] = []
dynamic_cat_cols[dataset] = ["concept:name"]
dynamic_num_cols[dataset] = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]

#### Bpic2012w Cases settings ####
datasets = ["bpic2012w_%s" % i for i in range(1, 3)]

for dataset in datasets:
    
    filename[dataset] = os.path.join(logs_dir, "%s.csv" % dataset)

    case_id_col[dataset] = "case:concept:name"
    activity_col[dataset] = "concept:name"
    resource_col[dataset] = "org:resource"
    timestamp_col[dataset] = "time:timestamp"
    label_col[dataset] = "label"
    magnitude_col[dataset] = "magnitude"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["concept:name", "org:resource"] # i.e. event attributes
    static_cat_cols[dataset] = []
    dynamic_num_cols[dataset] = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['case:AMOUNT_REQ']