# Bankruptcy-Classification
An evaluation of client's economic health and with the predominance of data collection, using ML algorithms to determine their ability to pay back their loans.

# Data

Data collected on Polish companies from 2000-2012. The data set consists of over 60 numerical attributes extracted from the financial statements of Polish companies.
More than 50% of observations have at least one missing value.
The dataset is highly unbalanced (bankruptcy and non-bankruptcy) with
approximately 4% and 96% , respectively.

# Training

```python
class_weights = {0 : 1, 1 : (len(y_train_) - sum(y_train_)) / sum(y_train_)}
best_params = []

logreg_params = {'C' : [1e-2, 1e-1, 1, 10, 100],
                 'class_weight' : ['balanced', class_weights]}

svm_params = {'C' : [1e-2, 1e-1, 1, 10, 100],
                'kernel' : ['rbf'],
                'class_weight' : ['balanced', class_weights]}

rf_params = {'n_estimators' : [50, 100, 200],
               'criterion' : ['gini', 'entropy'],
               'max_depth' : [3, 5, 7, None],
               'min_samples_leaf' : [1, 3, 5],
               'class_weight' : ['balanced', class_weights]}

adaboost_params = {'n_estimators' : [50, 100, 200]}

gradboost_params = {'n_estimators' : [50, 100, 200],
                    'max_depth' : [3, 5, 7, None],
                    'min_samples_leaf' : [1, 3, 5]}


clfs = [
        ('LogReg', LogisticRegression(), logreg_params),
        ('SVM', SVC(), svm_params),
        ('RF', RandomForestClassifier(), rf_params),
        ('AdaBoost', AdaBoostClassifier(), adaboost_params),
        ('GradBoosting', GradientBoostingClassifier(), gradboost_params)
]

for clf in clfs:
    print(clf[0])
    best_p = grid_search(clf[1], clf[2], X_app2, y_train_)
    print(best_p)
    best_params.append([clf[0], best_p])
```

