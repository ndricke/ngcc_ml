import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier

# Data Analysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


def group_split_evaluate(model, df_xy, feature_cols, target_col="Doesitbind", n_splits=10, group_col="Catalyst Name"):
    all_scores = []
    split_groups = GroupShuffleSplit(test_size=0.10, n_splits=n_splits, random_state = 7).split(df, groups=df[group_col])
    for train_inds, test_inds in split_groups:
        train = df_xy.iloc[train_inds]
        test = df_xy.iloc[test_inds]
        X_train_group = train[feature_cols]
        X_test_group = test[feature_cols]
        y_test_group = test[target_col]
        y_train_group = train[target_col]
        model.fit(X_train_group, y_train_group)
        score = model.score(X_test_group, y_test_group)
        all_scores.append(score)
        print('Accuracy of RFC on test set: {:.2f}'.format(score))
        print('Accuracy of RFC on training set: {:.2f}'.format(model.score(X_train_group, y_train_group)))
    print("mean:", np.mean(all_scores))
