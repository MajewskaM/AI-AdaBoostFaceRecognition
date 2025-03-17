import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Classifiers
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from adaboost import load_data


# get current working directory
working_directory = os.getcwd()
data_source_directory = os.path.join(working_directory, 'famous48')

no_faces = 48
X, y = load_data(data_source_directory, no_faces)

# Calculate the mean number of photos of each person (labels 0-47)
mean_photos_per_person = len(y) / no_faces
print("Mean number of photos of each person:", mean_photos_per_person)

# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)

print("Shape of training data: ", X_train.shape)
print("Shape of training label: ", y_train.shape)
print("Shape of test data: ", X_test.shape)
print("Shape of test label: ", y_test.shape)


# Preparing bar chart with accuarcies for whole dataset
# Using already calculated parameters of chosen classifiers
decision_tree = AdaBoostClassifier(estimator=DecisionTreeClassifier(
    max_depth=4, min_samples_leaf=5), learning_rate=0.5, n_estimators=500, algorithm='SAMME')

logistic_regression = AdaBoostClassifier(estimator=LogisticRegression(
    max_iter=1000, C=100), learning_rate=0.01, n_estimators=50, algorithm='SAMME')

random_forest = AdaBoostClassifier(estimator=RandomForestClassifier(
    max_depth=None, min_samples_leaf=1, min_samples_split=4), learning_rate=0.1, n_estimators=500, algorithm='SAMME')

linear_svc = AdaBoostClassifier(estimator=LinearSVC(
    C=100), learning_rate=0.01, n_estimators=500, algorithm='SAMME')

sgdc_classifier = AdaBoostClassifier(estimator=SGDClassifier(
    alpha=0.001), learning_rate=0.01, n_estimators=50, algorithm='SAMME')

base_classifiers = {logistic_regression: 'Logistic Regression', linear_svc: 'LinearSVC',
                    decision_tree: 'Decision Tree', random_forest: 'Random Forest', sgdc_classifier: 'SGD Classifier'}

fig, ax = plt.subplots()
accuracy_scores = []

# Calculating AdaBoost accuarcy for with each classifier on whole dataset
for classifier, name in base_classifiers.items():
    print("Classifier:", name)
    model = classifier

    # predict the class of test data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    overall_accuracy = accuracy_score(y_test, y_pred)
    print("Model Evaluation - accuracy:", overall_accuracy)
    print()
    accuracy_scores.append(overall_accuracy)

# Plotting the accuracy scores
x_labels = ['Logistic Regression', 'LinearSVC',
            'Decision Tree', 'Random Forest', 'SGD Classifier']
ax.bar(x_labels, accuracy_scores)
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Different Base Classifiers in AdaBoost')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
