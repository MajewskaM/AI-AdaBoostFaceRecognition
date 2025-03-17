import os
from adaboost import load_data, prepare_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Classifiers
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# perform grid search for given classifier, param_grid with ada boost
def grid_search_adaboost(classifier, classifier_name, param_grid, X_train, y_train, X_test, y_test):

    grid_search = GridSearchCV(AdaBoostClassifier(estimator=classifier, algorithm='SAMME'),
                               param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2).fit(X_train, y_train)

    print(f"Best parameters for {classifier_name}:", grid_search.best_params_)
    print(f"Best Score for {classifier_name}:", grid_search.best_score_)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    prepare_report(y_test, y_pred)


# get current working directory
working_directory = os.getcwd()
data_source_directory = os.path.join(working_directory, 'famous48')

# due to time complexity estimate best parameters for 10 faces
no_faces = 10
X, y = load_data(data_source_directory, no_faces)

# Calculate the mean number of photos of each person (labels 0-9)
mean_photos_per_person = len(y) / no_faces
print("Mean number of photos of each person:", mean_photos_per_person)

# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)

# accessing the shape of data & labels
print("Shape of training data: ", X_train.shape)
print("Shape of training label: ", y_train.shape)
print("Shape of test data: ", X_test.shape)
print("Shape of test label: ", y_test.shape)


#############################################
# GridSearch for DECISION TREE

parameters_dt = {'estimator__max_depth': [4],
                 'estimator__min_samples_leaf': [5, 10],
                 'n_estimators': [50, 300, 500],
                 'learning_rate': [0.01, 0.1, 0.5, 1]}

grid_search_adaboost(DecisionTreeClassifier(), "Decision Tree",
                     parameters_dt, X_train, y_train, X_test, y_test)


#############################################
# GridSearch for LINEAR SVC

parameters_svc = {'estimator__C': [0.01, 1, 10, 100],
                  'n_estimators': [50, 300, 500],
                  'learning_rate': [0.01, 0.1, 0.5, 1]}

grid_search_adaboost(LinearSVC(), "LinearSVC",
                     parameters_svc, X_train, y_train, X_test, y_test)


#############################################
# GridSearch for LOGISTIC REGRESSION

parameters_lr = {'estimator__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                 'n_estimators': [50, 300, 500],
                 'learning_rate': [0.01, 0.1, 0.5, 1]}

grid_search_adaboost(LogisticRegression(
    max_iter=1000), "Logistic Regression", parameters_lr, X_train, y_train, X_test, y_test)


#############################################
# GridSearch for RANDOM FOREST

parameters_rf = {'estimator__max_depth': [3, None],
                 'estimator__min_samples_leaf': [1, 2],
                 'estimator__min_samples_split': [2, 4],
                 'n_estimators': [50, 300, 500],
                 'learning_rate': [0.01, 0.1, 0.5, 1]}

grid_search_adaboost(RandomForestClassifier(), "Random Forest",
                     parameters_rf, X_train, y_train, X_test, y_test)


#############################################
# GridSearch for SGD CLASSIFIER

parameters_sgd = {'estimator__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
                  'n_estimators': [50, 300, 500],
                  'learning_rate': [0.01, 0.1, 0.5, 1]}

grid_search_adaboost(SGDClassifier(), "SGD Classifier",
                     parameters_sgd, X_train, y_train, X_test, y_test)
