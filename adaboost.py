
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.svm import LinearSVC

# load dataset from specified "source" directory
# extracting features and labels of categories


def load_data(source_directory, no_faces):
    features_list = []
    labels = []

    # iterate over all the files in the source directory
    for filename in os.listdir(source_directory):

        # for this project files with source data ends with: 'x24.txt'
        if filename.endswith("x24.txt"):
            new_file_dir = os.path.join(source_directory, filename)
            with open(new_file_dir, 'r') as f:
                num_examples = int(f.readline())
                num_pixels = int(f.readline())

                for _ in range(num_examples):
                    example = list(map(float, f.readline().split()))
                    label = int(example[num_pixels + 2])

                    if (no_faces <= 48):

                        # filter out labels of desired number of faces
                        if label <= no_faces - 1:
                            features = example[:num_pixels]
                            labels.append(label)
                            features_list.append(features)
                    else:
                        print("Error: There are only 48 faces in the given dataset")

    return np.array(features_list), np.array(labels)

# training classifiers and evaluating model


def evaluate_results(model, X_test, y_test, X_train, y_train, cv, name):
    # fitting the model to the training data
    model.fit(X_train, y_train)

    cv_train_score = cross_val_score(
        model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"On average, {name} model has f1 score of {cv_train_score.mean():3f} +/- {cv_train_score.std():.3f} on the training set.")

    # predict the class of test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy for each individual photo
    individual_accuracies = (y_pred == y_test)

    # Calculate accuracy per each person (label)
    label_accuracy = defaultdict(list)
    for i, accuracy in enumerate(individual_accuracies):
        label = y_test[i]
        label_accuracy[label].append(accuracy)
    label_accuracy_sorted = {label: accuracies for label, accuracies in sorted(
        label_accuracy.items(), key=lambda x: x[0])}

    # calculate accuracy per each person as a percentage
    accuracy_percentage_per_person = {}
    for label, accuracies in label_accuracy_sorted.items():
        total_photos = len(accuracies)
        correct_predictions = sum(accuracies)
        accuracy_percentage = (correct_predictions / total_photos) * 100
        accuracy_percentage_per_person[label] = accuracy_percentage

    print("Accuracy per each person (label) in percentage:")
    for label, accuracy_percentage in accuracy_percentage_per_person.items():
        print(f"Person {label}: {accuracy_percentage:.2f}%")

    # overall model accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)
    print("Model Evaluation - accuracy:", overall_accuracy)

    prepare_report(y_test, y_pred)

# preparing report about the model predictions accuracy


def prepare_report(y_test, y_pred):

    print("#Classification report")
    print(classification_report(y_test, y_pred))

    print("#Confusion matrix")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred))
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()


# get current working directory
working_directory = os.getcwd()
data_source_directory = os.path.join(working_directory, 'famous48')

no_faces = 48
X, y = load_data(data_source_directory, no_faces)

# Calculate the mean number of photos of each person (labels 0-47)
mean_photos_per_person = len(y) / no_faces
print("Mean number of photos of each person:", mean_photos_per_person)

# Initialize lists to store results
training_sizes = []
accuracies = []

# Vary training set sizes
# using LinearSVC as base estimator for AdaBoost as it provided best accuracy for whole dataset
for i in range(1, 10):
    train_size = i / 10.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=1)
    training_sizes.append(len(X_train))

    linear_svc = AdaBoostClassifier(estimator=LinearSVC(
        C=100), learning_rate=0.01, algorithm='SAMME')
    accuracy = evaluate_results(linear_svc, X_test, y_test, X_train, y_train, ShuffleSplit(
        n_splits=10, test_size=0.1, random_state=42), "LinearSVC")
    accuracies.append(accuracy)

# Plotting the results
plt.plot(training_sizes, accuracies, marker='o')
plt.title('Accuracy vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
