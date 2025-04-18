import json
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from train_model import read_csv_file, close_all_files

TESTING_SAMPLES = "./Dataset/X_test.csv"
TESTING_LABELS = "./Dataset/y_test.csv"
MODEL_PATH = "./model.json"

opened_files = []

def load_model( path : str ):
    model_file = open( path, "r" )
    model = json.load( model_file )

    opened_files.append( model_file )

    return model

def calculate_bayes( model : dict, sample, columns : list, label : str ):
    bayesion_product = 1
    
    for feature, column in zip( sample, columns ):
        try:
            conditional_prob = model[label][column][feature]
        except:
            conditional_prob = 1
        bayesion_product += math.log( conditional_prob )
    
    bayesion_product += math.log( model["prior"][label] )

    return bayesion_product

def predict_naive_bayes( model : dict, samples ):
    for sample in samples:
        columns = sample
        break

    predictions = []

    for sample in samples:
        true_prediction = calculate_bayes( model, sample, columns, label="True" )
        false_prediction = calculate_bayes( model, sample, columns, label="False" )

        if true_prediction >= false_prediction:
            predictions.append( "True" )
        else:
            predictions.append( "False" )

    return predictions

def evaluate_results( predictions, labels ):
    true_positive   = 0     # Predicted true, actually true
    false_positive  = 0     # Predicted true, actually false     
    true_negative   = 0     # Predicted false, actually false
    false_negative  = 0     # Predicted false, actually true

    for prediction, label in zip( predictions, labels ):
        label = label[0] # Due to the data being read from a csv file 

        if prediction == "True":
            if label == "True":
                true_positive += 1
            elif label == "False":
                false_positive += 1
        elif prediction == "False":
            if label == "False":
                true_negative += 1
            elif label == "True":
                false_negative += 1
    
    return ( true_positive, false_positive, true_negative, false_negative )

def plot_confusion_matrix(true_positive, false_positive, true_negative, false_negative):
    matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Positive", "Predicted Negative"],
                yticklabels=["Actual Positive", "Actual Negative"])
    
    plt.title("Confusion Matrix")
    plt.savefig( "confusion_matrix.png" )
    plt.show()


model = load_model( MODEL_PATH )
test_samples = read_csv_file( TESTING_SAMPLES )
test_labels = read_csv_file( TESTING_LABELS )

predictions = predict_naive_bayes( model=model, samples=test_samples )

true_positive, false_positive, true_negative, false_negative = evaluate_results( predictions, test_labels )

plot_confusion_matrix( true_positive, false_positive, true_negative, false_negative )

accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)
precision = true_positive / ( true_positive + false_positive )

print( f"Accuracy: {accuracy}\nPrecision: {precision}" )

close_all_files()