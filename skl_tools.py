import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import clone

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


def group_kfold_evaluate(model, df_xy, feature_cols, target_col="Doesitbind", n_splits=10, group_col="Catalyst Name"):
    all_scores, all_test = [], []
    split_groups = GroupKFold(n_splits=n_splits).split(df_xy[feature_cols], df_xy["Doesitbind"], df_xy[group_col])
    for train_inds, test_inds in split_groups:
        model_kfg = clone(model)
        train = df_xy.iloc[train_inds]
        test = df_xy.iloc[test_inds]
        X_train_group = train[feature_cols]
        X_test_group = test[feature_cols]
        y_test_group = test[target_col]
        y_train_group = train[target_col]
        model_kfg.fit(X_train_group, y_train_group)
        score = model_kfg.score(X_test_group, y_test_group)
        test = test.assign(Doesitbind_pred=model_kfg.predict(X_test_group))
        test = test.assign(Doesitbind_predproba=model_kfg.predict_proba(X_test_group)[:,1])
        all_test.append(test)
        all_scores.append(score)
        print('Accuracy of RFC on test set: {:.2f}'.format(score))
        print('Accuracy of RFC on training set: {:.2f}'.format(model_kfg.score(X_train_group, y_train_group)))
    print("mean:", np.mean(all_scores))
    df_pred_aug = pd.concat(all_test)
    return all_scores, df_pred_aug


def search_for_active_catalysts(df_catalysts, order_col, feature_cols, target_col="Doesitbind", find_num=10):
    """
    df_catalysts (pandas dataframe): catalysts to search
    order_col (str): column name to sort catalysts by. Expected for predict_proba or random values
    """
    df_sort = df_catalysts.sort_values(by=order_col, ascending=False)
    found_list = []
    count = 0
    for index, row in df_sort.iterrows():
        if row["Catalyst Name"] not in found_list:
            if row[target_col] == 1:
                found_list.append(row["Catalyst Name"])
            count += 1
            assert len(found_list) <= find_num
            if len(found_list) == find_num:
                break
    return found_list, count
