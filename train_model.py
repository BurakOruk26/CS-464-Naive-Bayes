import pandas
import csv
import json

TRAINING_SAMPLES = "./Dataset/X_train.csv"
TRAINING_LABELS = "./Dataset/y_train.csv"
SAVE_MODEL_PATH = "./model.json"
LABEL_COLUMN = "galaxy"

opened_files = []

def close_all_files():
    for file in opened_files:
        file.close()

def read_csv_file_pandas( path : str ):
    file = open( path, "r" )
    csv_file = pandas.read_csv( file )

    opened_files.append( file )

    return csv_file

def read_csv_file( path : str ):
    file = open( path, "r" )
    csv_file = csv.reader( file )

    opened_files.append( file )

    return csv_file

def save_model( model : dict, path : str ):
    # Creates the target file if it doesn't exist
    with open( path, "a+" ):
        pass

    file = open( path, "w" )
    json.dump( model, file, indent=3 )

    opened_files.append( file )

def convert_labels_to_list( labels ):
    label_list = []
    for label in labels:
        label = label[0]
        label_list.append( label )
    
    return label_list

def add_labels( data: pandas.DataFrame, labels ):
    return data.assign( galaxy = labels ) # This part is hardcoded. If it is to be changed, the variable LABEL_COLUMN also should be.

def laplace_smoothing( data: pandas.DataFrame, features : list ):
    sample_features_true = {}
    sample_features_false = {}

    for i in range( len(features) - 1 ):
        feature = features[i]
        sample_features_true[ feature ] = {}
        sample_features_false[ feature ] = {}
        samples = data[ feature ]
        samples = list( set( samples ) ) # Find unique items
        for sample in samples:
            sample_features_true[ feature ][ sample ] = 1
            sample_features_false[ feature ][ sample ] = 1
    
    initial_model = {
        "True": sample_features_true,
        "False" : sample_features_false
    }

    return initial_model

def calculate_prior( data : pandas.DataFrame ):
    labels = data[ LABEL_COLUMN ].tolist()

    true_count = 0
    false_count = 0

    for label in labels:
        if label == "True":
            true_count += 1
        if label == "False":
            false_count += 1

    prior = {
        "True" : true_count / (true_count + false_count),
        "False" : false_count / (true_count + false_count)
    }
    return prior

def calculate_feature_prob( data: pandas.DataFrame, feature : str, label : str, sample_features : dict ):
    for sample_feature, sample_label in zip( data[ feature ], data[ LABEL_COLUMN ] ):
        if sample_label != label:
            continue

        if sample_feature not in sample_features:
            sample_features[ sample_feature ] = 1
        else:
            sample_features[ sample_feature ] += 1

    total_count = 0
    for sample_feature_count in sample_features.values():
        total_count += sample_feature_count

    for sample_feature in sample_features:
        sample_features[sample_feature] /= total_count

def train_model( data : pandas.DataFrame, model : dict, features : list, labels : list ):
    prior = calculate_prior( data=data )
    model["prior"] = prior

    labels = [ "True", "False" ]
    for label in labels:
        for i in range( len(features) - 1 ):
            feature = features[i]
            sample_features = model[label][feature]
            calculate_feature_prob( data=data, feature=feature, label=label, sample_features=sample_features )

    return model

# ---------- #
# -- Main -- #
# ---------- #

# Read the training data and its labels
training_data = read_csv_file_pandas( TRAINING_SAMPLES )
training_label = read_csv_file( TRAINING_LABELS )

# Read the labels as a list and add to the data as a column
training_label_list = convert_labels_to_list( training_label )
training_data = add_labels( training_data, training_label_list )

# Extract the feature names for calculations
features = training_data.columns

initial_model = laplace_smoothing( data=training_data, features=features )

"""
aa = json.dumps( initial_model, indent=3 )
print(aa)
close_all_files()
exit()
"""

model = train_model( data=training_data, model=initial_model, features=features, labels=["True","False"] )

save_model( model=model, path=SAVE_MODEL_PATH )

close_all_files()
