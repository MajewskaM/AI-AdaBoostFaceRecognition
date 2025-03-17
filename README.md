# AI-AdaBoostFaceRecognition
University project for the Artificial Intelligence course. The aim of the project is to use adaboost selected multi-class variant classifier for face recognition on Famous48 dataset.

## Concept & Approach
The project is based on the core principle of AdaBoost, which involves fitting a sequence of weak learners (e.g., base classifiers like decision trees) on repeatedly re-sampled versions of the data. Each sample is assigned a weight that is adjusted after each training step so that misclassified samples receive higher weights. Samples with higher weights have a greater chance of being selected multiple times in the new data set. <br><br>
Additionally, the project compares the accuracy of the boosted classifiers and provides the best obtained results. The results were obtained using three files:
- [_grid_search.py_](https://github.com/MajewskaM/AI-AdaBoostFaceRecognition/blob/main/grid_search.py)
- [_classifiers_comparison.py_](https://github.com/MajewskaM/AI-AdaBoostFaceRecognition/blob/main/classifiers_comparison.py)
- [_adaboost.py_](https://github.com/MajewskaM/AI-AdaBoostFaceRecognition/blob/main/adaboost.py)

## Functions developed
1. **load_data(source_directory, no_faces)**
  - Loads dataset, extracting features and labels. 
  - Filters data based on the specified number of faces.

2. **evaluate_results(model, X_test, y_test, X_train, y_train, cv, name)**
  - Trains the model and evaluates it using cross-validation.
  - Prints classification report and confusion matrix.
  - Calculates individual accuracies per person.

3. **prepare_report(y_test, y_pred)**
  - Prints classification report and plots confusion matrix.

4. **grid_search_adaboost(classifier, classifier_name, param_grid, X_train, y_train, X_test, y_test)**
  - Performs grid search for optimal parameters of AdaBoost with specified classifier.
  - Displays best parameters and scores.
  - Prints classification report and confusion matrix.

## Customizing the model: grid_search.py
File is responsible for customizing hyperparameters of machine learning models to given dataset. Approach is to objectively search different values for model hyperparameters and choose a subset that results in a model that achieves the best performance on a given dataset. This is called **hyperparameter optimization** or **hyperparameter tuning** and is available in the scikit-learn Python machine learning library. 
Due to huge time complexity, **GridSearchCV** function is limited with cross validation and use first 10 faces from given dataset.
Description of Actions
1. Loading dataset using [**load_data**](##functions-developed) function from adaboost.py to extract features and labels.
2. Splitting data into training and testing sets - **train_test_split** function.
3. Performs grid search for optimal parameters for **Decision Tree, Linear SVC, Logistic Regression, Random Forest, and SGD Classifier -** [**grid_search_adaboost**](##functions-developed) function.
4. Displaying best parameters and scores for each classifier. **(.fit, .predict, [prepare_report](##functions-developed))**

## Fitting parameters for 10 faces
```
Mean number of photos of each person: 137.9
Shape of training data:  (1241, 576)
Shape of training label:  (1241,)
Shape of test data:  (138, 576)
Shape of test label:  (138,)
```

1.	DecisionTreeClassifier
<img src="https://github.com/user-attachments/assets/b343ae1e-bafe-430d-a669-e0a365c49b99" alt="DecisionTree" width="400" height="300">
<br>

```
Fitting 5 folds for each of 8 candidates, totalling 40 fits

Best parameters for DecisionTree: {'estimator__max_depth': 4, 'estimator__min_samples_leaf': 5,
'learning_rate': 0.5, 'n_estimators': 500}
Best Score for DecisionTree: 0.828381267003498
```

2.	LogisticRegression
<img src="https://github.com/user-attachments/assets/9842fd84-9c7a-4261-b034-bc1a97b0944f" alt="LogisticRegression" width="400" height="300">
<br>

```
Fitting 5 folds for each of 48 candidates, totalling 240 fits 

Best parameters for LogisticRegression: {'estimator__C': 100, 'learning_rate': 0.01, 'n_estimators': 50}
Best Score for LogisticRegression: 0.894448762793108
```

3.	RandomForestClassifier
<img src="https://github.com/user-attachments/assets/fd10dad5-ca1f-4118-8cd1-52dae21a5928" alt="LogisticRegression" width="400" height="300">
<br>

```
Fitting 5 folds for each of 48 candidates, totalling 240 fits
 
Best parameters for RandomForestClassifier: {'estimator__max_depth': None, 'estimator__min_samples_leaf': 1,
 'estimator__min_samples_split': 4, 'learning_rate': 0.1, 'n_estimators': 500}
Best Score for RandomForestClassifier: 0.8478260869565217
```

4.	LinearSVC
<img src="https://github.com/user-attachments/assets/360d16e2-21f7-47e8-a08c-7186f94fec7e" alt="LogisticRegression" width="400" height="300">
<br>

```
Fitting 5 folds for each of 16 candidates, totalling 80 fits        

Best parameters for LinearSVC: {'estimator__C': 100, 'learning_rate': 0.01, 'n_estimators': 500}
Best Score for LinearSVC: 0.9057971014492754
```


5.	SGDClassifier
<img src="https://github.com/user-attachments/assets/313b8f3c-df22-4fcb-980f-47a265ca0ce9" alt="LogisticRegression" width="400" height="300">
<br>

```
Fitting 5 folds for each of 168 candidates, totalling 840 fits

Best parameters for SGDClassifier: {'estimator__alpha': 0.001, 'estimator__penalty': 'l2',
 'learning_rate': 0.01, 'n_estimators': 500}
Best Score for SGDClassifier: 0.8985507246376812
```

Based on these results for given 10 faces, we will use the **LinearSVC classifier** for our dataset as it delivered the best performance – about  **90% accuracy**.
The confusion matrixes shows that across all evaluated classifiers, the best accuracy is consistently obtained for person no. 8, while the worst accuracy is consistently for person no. 9. Among the classifiers, the DecisionTreeClassifier, used as the default estimator for AdaBoost, performed the worst overall. 


## classifiers_comparison.py
1. Loading dataset using [**load_data**](##functions-developed) function from adaboost.py to extract features and labels.
2. Splitting data into training and testing sets - **train_test_split** function.
3. Classifier Comparison - training various classifiers (AdaBoost with different base classifiers) on the training set. **(.fit)** 							     4. Evaluates accuracy of different classifiers on the whole famous48 dataset **(.predict, accuracy_score)**.
5. Plotting accuracy of different base classifiers in AdaBoost. **(matplotlib.pyplot)**

### Products: Accuracy of a different AdaBoost base classifiers on the whole dataset
![AdaBoostClassifiersWholeDataset](https://github.com/user-attachments/assets/bd1b0b3e-1424-40c7-9b50-34a2d8fb56f1)

```
Mean number of photos of each person: 142.39583333333334
Shape of training data:  (6151, 576)
Shape of training label:  (6151,)
Shape of test data:  (684, 576)
Shape of test label:  (684,)
```
```
Classifier: Logistic Regression
Model Evaluation - accuracy: 0.7412280701754386
```

```
Classifier: LinearSVC
Model Evaluation - accuracy: 0.8245614035087719
```

```
Classifier: Decision Tree
Model Evaluation - accuracy: 0.47514619883040937
```

```
Classifier: Random Forest
Model Evaluation - accuracy: 0.6871345029239766
```

```
Classifier: SGD Classifier
Model Evaluation - accuracy: 0.7178362573099415 
```

For given dataset the best accuracy is obtained with LinearSVC classifier with parameters shown below. 
```
AdaBoostClassifier(estimator = LinearSVC(C=100), learning_rate = 0.01, algorithm='SAMME')
```

## adaboost.py
1. Loading dataset using [**load_data**](##functions-developed) function from adaboost.py to extract features and labels.
2. Model Training - fitting models to training data (**train_test_split**).
3. Evaluation using cross-validation, displaying accuracy per person and overall model accuracy [**(evaluate_results, prepare_report)**](##functions-developed).
4. Varying Training Set Sizes from 10% to 90% and evaluating performance (repeating points 2 & 3).
5. Plotting accuracy vs training set size of Adaboost with **LinearSVC (matplotlib.pyplot).**

### Products: Accuracy vs Training set size comparison

 ![Accuracy vs Trainning set size](https://github.com/user-attachments/assets/28cb440d-5a7d-4458-b5b6-eeed62cfd383)

### Results: Performance on the 48 faces (dataset)
```
Mean number of photos of each person: 683.5
Shape of training data:  (6151, 576)
Shape of training label:  (6151,)
Shape of test data:  (684, 576)
Shape of test label:  (684,)

On average, LinearSVC model has f1 score of 0.805533 +/- 0.015 on the training set.
```
Accuracy per each person (label) in percentage:
<br>
![Accuracy per each person (label)](https://github.com/user-attachments/assets/a142ff0f-b5f5-40c4-ae16-a0ad0122598c)

### Model Evaluation

The summary of accuracies obtained by the hyperparameter-tuned classifiers on the entire dataset reveals that the LinearSVC classifier using 90% of the dataset as a training size performed the best. 

## Libraries & build-in functions used in project
- os – operating system dependent functionality such as reading or writing to the file system
  - os.getcwd() - returns the current working directory as a string.
  - os.path.join() - joins one or more path components intelligently.
- numpy – numerical computing in Python, support for arrays, matrices, and many mathematical functions
- collections – alternatives to Python’s general-purpose built-in containers like dict, list etc.
  - defaultdict() - A subclass of the built-in dict class, provides a default value for the dictionary when a key does not exist.
- matplotlib.pyplot – Plotting results
- sklearn – Machine learning library, providing simple and efficient tools for data mining and data analysis (classifiers, metrics, model selection tools)
  - sklearn.tree – DecisionTreeClassifier()
    - max_depth: the maximum depth of the tree, helps prevent overfitting by restricting the complexity of the model.
    - min_samples_leaf: specifies the minimum number of samples required to be at a leaf node, can also help prevent overfitting.
- sklearn.ensemble – AdaBoostClassifier()
  - n_estimators: defines the number of boosting stages to be run, how many weak learners (trees) should be combined to form the final strong learner.
  - learning_rate: contribution of each tree by the learning rate, lower values make the model more robust to overfitting but may require more boosting stages.
  - algorithm: SAMME discrete boosting algorithm.

- sklearn.ensemble – RandomForestClassifier()
  - max_depth: Controls the maximum depth of the tree.
  - min_samples_leaf: The minimum number of samples required to be at a leaf node.
  - min_samples_split: The minimum number of samples required to split an internal node.

- sklearn.svm – LinearSVC(), class supports both dense and sparse input and the multiclass support is handled according to a one-vs-the-rest scheme
  - C: regularization parameter of the LinearSVC, where higher values correspond to less regularization.

- sklearn.linear_model – SGDClassifier()
  - alpha: Constant that multiplies the regularization term. The higher the value, the stronger the regularization.

- sklearn.linear_model – LogisticRegression()
  - C: inverse of regularization strength.

- sklearn.model_selection – train_test_split()
  - test_size/train_size: represent the proportion of the dataset to include in the test/train split.
  - random_state: controls the shuffling applied to the data before applying the split.

- sklearn.model_selection – ShuffleSplit()
  - n_splits: number of re-shuffling & splitting iterations.
  - test_size/train_size: represent the proportion of the dataset to include in the test/train split.
  - random_state: controls the shuffling applied to the data before applying the split.

- sklearn.model_selection – GridSearchCV()
  - estimator: assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a score function, or scoring must be passed.
  - param_grid: dictionary with parameters names (str) as keys and lists of parameter settings to try as values, enables searching over any sequence of parameter settings.
  - scoring: strategy to evaluate the performance of the cross-validated model on the test set.
  - n_jobs: number of jobs to run in parallel, -1 means using all processors
  - verbose: controls the verbosity: the higher, the more messages.
  - Implements a “fit” and a “score” method.

- Evaluating the results:
  - sklearn.model_selection – cross_val_score()
  - sklearn.metrics – accuracy_score()
  - sklearn.metrics – classification_report(), confusion_matrix(), ConfusionMatrixDisplay()
Also used:
- model.fit(X, y): Trains the model using the provided features (X) and labels (y).
- model.predict(X): Predicts the labels for the provided features (X) based on the trained model.
- 
