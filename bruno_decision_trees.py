#!/usr/bin/env python
# coding: utf-8

# # Decision Trees
# Supervised Learning Sec. 001</br>
# Student: Bruno Morgado</br>
# Student # 301-154-898


# Import necessary modules
import numpy as np
import pandas as pd
import time
import math
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os

# Helper function
# Where to save the figures
PROJECT_ROOT_DIR = "."
ASSIGNMENT = "decision_trees"

def image_path(fig_id):
    path = os.path.join(PROJECT_ROOT_DIR, "images", ASSIGNMENT)
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.join(path, fig_id)
   

# Helper Function to plot confusion Matrix
def plot_confusion_matrix(confusion_matrix, y_limit: list, color_map: str):
    #Plot the confusion Matrix
    fig, ax = plt.subplots(figsize=(10,6))
    title = f'Confusion matrix: Decision-Trees'
    # create heatmap
    sns.heatmap(confusion_matrix, annot = True, cmap = color_map ,fmt='g')
    ax.xaxis.set_label_position("top")
    ax.set_ylim(y_limit)
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])
    plt.title(title, fontsize=20, pad=10.0)
    plt.ylabel('Actual label', fontsize='large')
    plt.xlabel('Predicted label', fontsize='large')
    plt.tight_layout()

# Load student-por.csv
data_bruno = pd.read_csv('student-por.csv', sep=';')

# Show all columns in the dataset
pd.set_option('display.max_columns', None)

# Explore the first 5 rows of the dataset
data_bruno.head()

# Explore the last 5 rows
data_bruno.tail()

# Get the list of all columns
data_bruno.columns

data_bruno.info()

#Use a heatmap to visualize missing data
sns.set(rc={"figure.figsize":(14, 7)})
sns.heatmap(data_bruno.isna(),yticklabels=False,cbar=False,cmap='viridis')
plt.xticks(rotation = 45)
plt.show()


# `The dataset seems to be integral, without any missing values.
# In any case, columns with datatype 'object' need to be further
# investigated.`

# Descriptive statistics
data_bruno.describe()

# Select categorical data
data_bruno_categorical = data_bruno.select_dtypes(include=['object'])

# First five rows of the categorical sub-dataset
data_bruno_categorical.head()

# Numeric columns
data_bruno_numeric = data_bruno.drop(data_bruno_categorical.columns, axis = 1)

# First five rows of the numeric sub_dataset
data_bruno_numeric.head()

# Get unique values and their count for each categorical column
for col in data_bruno_categorical.columns:
    ticks = (int)((30 - len(col)) / 2)
    print(f'\n{ticks * "*"} {col} {ticks * "*"}')
    print(data_bruno_categorical[col].value_counts())

# Visual inspection 
nr_rows = 6
nr_cols = 3

cols_categories = data_bruno_categorical.columns
idx = 0

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3), squeeze=False)

for r in range(0,nr_rows):
    for c in range(0, nr_cols):  
        col = r*nr_cols+c
        if col < len(cols_categories):
            x = data_bruno_categorical[cols_categories[idx]].unique()
            y = data_bruno_categorical[cols_categories[idx]].value_counts()/len(data_bruno_categorical[cols_categories[idx]])
            idx += 1
            sns.set(style="darkgrid")
            sns.barplot(x = x, y = y, alpha=0.9,ax = axs[r][c])
            plt.ylabel('Number of Occurrences', fontsize=12)
            plt.xlabel(col, fontsize=12)
plt.tight_layout()    
plt.show()

# Instantiate a make_column_transformer to transform the categorical variables
# and leave the other columns as they are
categorical_cols = data_bruno_categorical.columns
transformer = make_column_transformer(
    (OneHotEncoder(), categorical_cols),
    remainder='passthrough',
    verbose_feature_names_out=False)

# Combine the three target variables into one categorical dependet variable
data_bruno['pass_bruno'] = np.where(data_bruno[['G1', 'G2', 'G3']].sum(axis=1) >= 35, 1, 0)

# Drop the original output variables
data_bruno.drop(['G1', 'G2', 'G3'], axis = 1, inplace = True)

# Inspect the transformed dataset
data_bruno.head()

# Store the dataset in .csv format
# data_transformed.to_csv('transformed.csv')

# Separate features from target variables
features_bruno = data_bruno.drop('pass_bruno', axis = 1)
target_variable_bruno = data_bruno['pass_bruno']

features_bruno.info()

target_variable_bruno.info()

# Number of unique values in the target variables
target_variable_bruno.value_counts()

proportion = target_variable_bruno.value_counts()/len(target_variable_bruno)

# Check for imbalanced target variables
plt.figure(figsize = (10,6))
sns.barplot(x = [0, 1], y = proportion)
plt.xticks(np.arange(2),('Pass = True', 'Pass = False'))
plt.ylabel('Proportion')
plt.show()

# Instantiate the decision tree classifier
clf_bruno = DecisionTreeClassifier(criterion='entropy', max_depth=5)

# Create a pipeline to streamline the transformation and prediction process
pipeline_bruno = Pipeline([
    ('col_transformer', transformer),
    ('clf_bruno', clf_bruno)
    ])

# Split the dataset into training and test datasets
X_train_bruno, X_test_bruno, y_train_bruno, y_test_bruno = train_test_split(features_bruno, target_variable_bruno, test_size=0.20, random_state=98)

# Fit the pipeline onto the training data
pipeline_bruno.fit(X_train_bruno, y_train_bruno)

# Get 5-fold cross-validation scores on the training datase
scores = cross_val_score(pipeline_bruno,
                        X_train_bruno,
                        y_train_bruno,
                        cv=10,
                        n_jobs=-1,
                        verbose=1)

print(scores)

# Get the mean accuracy score
print(scores.mean())

# Export a visual of the Decision Tree
export_graphviz(
clf_bruno,
out_file=image_path("student_por_tree.dot"),
feature_names=transformer.get_feature_names_out(),
class_names = ['0', '1'],
rounded=True,
filled=True
)

# ![student_por_tree.png](attachment:student_por_tree.png)

print(f'Training Score: {round(pipeline_bruno.score(X_train_bruno, y_train_bruno) * 100, 4)}%')

print(f'Test Score: {round(pipeline_bruno.score(X_test_bruno, y_test_bruno) * 100, 4)}%')

# Make predictions
y_pred = pipeline_bruno.predict(X_test_bruno)

cm = confusion_matrix(y_test_bruno, y_pred)

# Plot Confusion Matrix
plot_confusion_matrix(cm, [0,2], 'PuBu')

print(classification_report(y_test_bruno, y_pred))


# precision means what percentage of the positive predictions made were actually correct.
# 
# `TP/(TP+FP)`
# 
# Recall in simple terms means, what percentage of actual positive predictions were correctly classified by the classifier.
# 
# `TP/(TP+FN)`
# 
# F1 score can also be described as the harmonic mean or weighted average of precision and recall.
# 
# `2x((precision x recall) / (precision + recall))`

# ## Fine-tuning the model

parameters={'clf_bruno__min_samples_split' : range(10,300,20),
            'clf_bruno__max_depth': range(1,30,2),
            'clf_bruno__min_samples_leaf':range(1,15,3)}

print(parameters)

# Create a RandomizedSearchCV object
grid_search_bruno = RandomizedSearchCV(estimator = pipeline_bruno,
                                       param_distributions = parameters,
                                       scoring = 'accuracy',
                                       cv = 5,
                                       n_iter = 7,
                                       refit = True,
                                       n_jobs = -1,
                                       verbose = 3)

# Fitting RandomizedSearch onto the the training data
grid_search_bruno.fit(X_train_bruno, y_train_bruno)

# Best hyperparameters
print("tuned hpyerparameters :(best parameters) \n", grid_search_bruno.best_params_)

# Store the best model
best_model = grid_search_bruno.best_estimator_

best_model.score(X_train_bruno, y_train_bruno)

print(best_model)

# Make new predictions with the tuned model
final_pred = best_model.predict(X_test_bruno)

best_model.score(X_test_bruno, y_test_bruno)

# Print the classification Report
print('\t\tClassification Report\n\n',classification_report(y_test_bruno, final_pred))

# Import joblib to save the model
import joblib

joblib.dump(best_model, "DT_model.pkl")

joblib.dump(pipeline_bruno, "full_pipeline.pkl")

import dill

# Store the session
dill.dump_session('notebook_env.db')

# ## Additional test with - Cost Complexity Pruning

# Adding ccp_alpha to the parameters grid
parameters_pruning={'clf_bruno__min_samples_split' : range(10,300,20),
                    'clf_bruno__max_depth': range(1,30,2),
                    'clf_bruno__min_samples_leaf':range(1,15,3),
                    'clf_bruno__ccp_alpha': np.arange(0, 1.005, 0.005)}

# Create a RandomizedSearchCV object
grid_search_bruno_pruning = RandomizedSearchCV(estimator = pipeline_bruno,
                                       param_distributions = parameters_pruning,
                                       scoring = 'accuracy',
                                       cv = 5,
                                       n_iter = 10000,
                                       refit = True,
                                       n_jobs = -1,
                                       verbose = 3)

# Fitting RandomizedSearch onto the the training data
grid_search_bruno_pruning.fit(X_train_bruno, y_train_bruno)

# Store the best model
best_model_pruning = grid_search_bruno_pruning.best_estimator_

print(best_model_pruning)

# Make new predictions with the tuned model
pruning_pred = best_model_pruning.predict(X_test_bruno)

# Print the classification Report
print('\t\tClassification Report\n\n',classification_report(y_test_bruno, pruning_pred))

# Store the session
dill.dump_session('notebook_env_pruning.db')
