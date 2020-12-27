import pandas as pd
import numpy as np
import os
import re
import sys
import matplotlib.pyplot as plt
import seaborn
import rdkit
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


#regression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from math import sqrt
from sklearn.gaussian_process import GaussianProcessRegressor


from itertools import product


def process_data(df_in, oneHotCols=[], scaledCols=[]):
    df = df_in.copy()
    scaler = StandardScaler()
    oneHotEncoder = OneHotEncoder(categories = "auto", sparse = False)
    if scaledCols != []:
        df[scaledCols] = scaler.fit_transform(df[scaledCols])
    #if oneHotCols != []:
    #    df[encoded_columns] = oneHotEncoder.fit_transform(alldata[oneHotCols])
    return df


def plot_coefficients(classifier, feature_names, top_features=5):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 8))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(1, (len(feature_names)+1)), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2*top_features), feature_names[top_coefficients], rotation=20, ha='right', fontsize = 10)
    plt.show()

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    np.set_printoptions(precision=2)
    plt.show()


def plotData(X, y, dtc, f1, f2):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = dtc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

   # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired, s = 80)
    plt.xlabel(f1)
    plt.ylabel(f2)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

    
def getDataFromCSV(csv):
    df = pd.read_csv(csv)
    return (df)

def processData(alldata, features):
    cols = []
    scaledCols = []
    oneHotCols = []
    alreadyProcessedCols = []
    for feature in features:
        if feature != None:
            if feature == "NumberOfHydrogens":
                oneHotCols.append(feature)
                cols.append(feature)
            elif feature in ["OrthoOrPara", "Meta", "FartherThanPara", "IsInRingSize6", "IsInRingSize5"]:
                alreadyProcessedCols.append(feature)
                cols.append(feature)
            else:
                scaledCols.append(feature)
                cols.append(feature)
    
    print(scaledCols)
    print(oneHotCols)
    print(alreadyProcessedCols)
    
    
    scaler = StandardScaler()
    oneHotEncoder = OneHotEncoder(categories = "auto", sparse = False)
    if scaledCols != []:
        scaled_columns = scaler.fit_transform(alldata[scaledCols])
    else:
        scaled_columns = []
    #print(scaled_columns)
    encoded_columns = oneHotEncoder.fit_transform(alldata[oneHotCols])
    #print(encoded_columns)
    already_processed_columns = alldata[alreadyProcessedCols]
    already_processed_columns = already_processed_columns.values
    processed_data = np.concatenate((scaled_columns, already_processed_columns, encoded_columns), axis = 1)
    active_sites = alldata["Atom Number"].values
    catalyst_names = alldata["Catalyst Name"].values
    true_labels = alldata["Doesitbind"].values
    true_labels = true_labels.astype('int')
    df = pd.DataFrame(processed_data)
    df.insert(0,'Label', true_labels)
    df.insert(0,'ActiveSite',active_sites)
    df.insert(0,'CatalystName', catalyst_names)
    return(df)
    
def classify(alldata, which, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, c, poly, onehot, 
             confusion, plotcoef, checkmissed, testdata):
    
    #print(alldata.loc[(alldata["SpinDensity"]>0.27)  & (alldata["Doesitbind"]==False)])
    #print(alldata)
    testcolumn = alldata['Doesitbind']
    trues = 0
    falses = 0
    for value in testcolumn:
        if value == True:
            trues +=1
        else:
            falses +=1
    
    print("The number of binding active sites is ", trues)
    print("The number of nonbinding active sites is ", falses)
    
    cols = []
    scaledCols = []
    oneHotCols = []
    alreadyProcessedCols = []
    features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    processed_data = processData(alldata, features)
    if (testdata is not None):
        processed_test_data = processData(testdata, features)
    for feature in features:
        if feature != None:
            cols.append(feature)    
    
    print("For the features ", cols)

    X=processed_data
    y=alldata['Doesitbind'].astype('int')
    
    X_data = X.iloc[:,3:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train_data = X_train.iloc[:,3:]
    X_test_data = X_test.iloc[:,3:]

    
    logregression = LogisticRegression(solver = 'lbfgs')
    logregression.fit(X_train_data, y_train)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logregression.score(X_test_data, y_test)))
    print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logregression.score(X_train_data, y_train)))
    scores = cross_val_score(logregression, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #plotData(X, y, logregression)
    
    dtc = tree.DecisionTreeClassifier(max_depth = 2)
    dtc.fit(X_train_data, y_train)
    print('Accuracy of decision tree classifier on test set: {:.2f}'.format(dtc.score(X_test_data, y_test)))
    print('Accuracy of decision tree classifier on training set: {:.2f}'.format(dtc.score(X_train_data, y_train)))
    scores = cross_val_score(dtc, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    
    mlp = MLPClassifier(max_iter=2500, hidden_layer_sizes = (2048,15))
    mlp.fit(X_train_data, y_train)
    print('Accuracy of MLP classifier on test set: {:.2f}'.format(mlp.score(X_test_data, y_test)))
    print('Accuracy of MLP classifier on training set: {:.2f}'.format(mlp.score(X_train_data, y_train)))
    scores = cross_val_score(mlp, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    linsvc = LinearSVC(C = float(c))
    linsvc.fit(X_train_data, y_train)
    print('Accuracy of LinearSVC classifier on test set: {:.2f}'.format(linsvc.score(X_test_data, y_test)))
    print('Accuracy of LinearSVC classifier on training set: {:.2f}'.format(linsvc.score(X_train_data, y_train)))
    scores = cross_val_score(linsvc, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))   
    
    svc = SVC(C = float(c), kernel = 'rbf', gamma = 'scale')
    svc.fit(X_train_data, y_train)
    print('Accuracy of SVC on test set: {:.2f}'.format(svc.score(X_test_data, y_test)))
    print('Accuracy of SVC on training set: {:.2f}'.format(svc.score(X_train_data, y_train)))
    scores = cross_val_score(svc, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    
    rfc = RandomForestClassifier(n_estimators=1000, class_weight = {0:0.00000002, 1:0.99999998})
    rfc.fit(X_train_data, y_train)
    print('Accuracy of RFC on test set: {:.2f}'.format(rfc.score(X_test_data, y_test)))
    print('Accuracy of RFC on training set: {:.2f}'.format(rfc.score(X_train_data, y_train)))
    scores = cross_val_score(rfc, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    
    y_pred = mlp.predict(X_test_data)
    
    optimized_data = pd.DataFrame(data=X_test)
    optimized_data.insert(3,'pred', y_pred)

    if (testdata is not None):
        for_prediction = processed_test_data.iloc[:, 3:]
        test_pred = mlp.predict(for_prediction)
        processed_test_data['pred'] = test_pred
        successes = 0
        failures = 0
        correct_ids = 0
        for index, row in processed_test_data.iterrows():
            for i, r in optimized_data.iterrows():
                if r['CatalystName']==row['CatalystName']:
                    if r['ActiveSite']==row['ActiveSite']:
                        if r['Label'] == row['pred']:
                            correct_ids += 1
                            print('Correct id!')
                        if r['pred'] == row['pred']:
                            successes += 1
                            print("Success!")
                        else:
                            failures += 1
                            print("Failure")
        print("Total matches between optimized data and forcefield data: ", successes, "out of ", successes+failures)
        print("Total correct identifications: ", correct_ids, "out of ", successes+failures)

    classifier_dict = {"dtc" : dtc, "log" : logregression, "svc" : svc, "mlp" : mlp, "linsvc" : linsvc}


    if len(cols) == 2 and which != None:
        plotData(X_test_data.values[:,:], y_test, classifier_dict[which], f1, f2)
    
    if confusion!=0:
        plot_confusion_matrix(y_test, y_pred) 
    
    if plotcoef!=0:
        plot_coefficients(linsvc, cols)


def plotOverpotentialClassification(X, y,clf1, clf2, clf3, eclf):
    print(X)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

    for idx, clf, tt in zip(product([0, 1], [0, 1]),
                            [clf1, clf2, clf3, eclf],
                            ['MLP', 'Random forest',
                             'Linsvc', 'Logistic regression']):

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                      s=20, edgecolor='k')
        axarr[idx[0], idx[1]].set_title(tt)

    plt.show()

def classify_overpotential(alldata, which, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, c, poly, onehot, 
             confusion, plotcoef, checkmissed, testdata):
    
    #print(alldata.loc[(alldata["SpinDensity"]>0.27)  & (alldata["Doesitbind"]==False)])
    #print(alldata)
    testcolumn = alldata['activity']
    inactives = 0
    barely_actives = 0
    weakly_actives = 0
    actives = 0
    strongly_actives = 0
    best = 0
    for value in testcolumn:
        if value == 0:
            inactives +=1
        elif value == 1:
            barely_actives +=1
        elif value == 2:
            weakly_actives +=1
        elif value == 3:
            actives += 1
        elif value == 4:
            strongly_actives += 1
        elif value == 5:
            best += 1
        else:
            print("Error in test column")
    
    print("The number of inactive sites is ", inactives)
    print("The number of barely active sites is ", barely_actives)
    print("The number of weakly active sites is ", weakly_actives)
    print("The number of active sites is ", actives)
    print("The number of strongly active sites is ", strongly_actives)
    print("The number of best catalysts is ", best)
    
    cols = []
    scaledCols = []
    oneHotCols = []
    alreadyProcessedCols = []
    features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    processed_data = processData(alldata, features)
    if (testdata is not None):
        processed_test_data = processData(testdata, features)
    for feature in features:
        if feature != None:
            cols.append(feature)    
    
    print("For the features ", cols)

    X=processed_data
    y=alldata['activity'].astype('int')

    X_data = X.iloc[:,3:]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train_data = X_train.iloc[:,3:]
    X_test_data = X_test.iloc[:,3:]

    
    logregression = LogisticRegression(solver = 'lbfgs')
    logregression.fit(X_train_data, y_train)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logregression.score(X_test_data, y_test)))
    print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logregression.score(X_train_data, y_train)))
    scores = cross_val_score(logregression, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #plotData(X, y, logregression)
    
    dtc = tree.DecisionTreeClassifier(max_depth = 2)
    dtc.fit(X_train_data, y_train)
    print('Accuracy of decision tree classifier on test set: {:.2f}'.format(dtc.score(X_test_data, y_test)))
    print('Accuracy of decision tree classifier on training set: {:.2f}'.format(dtc.score(X_train_data, y_train)))
    scores = cross_val_score(dtc, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    
    mlp = MLPClassifier(max_iter=2500, hidden_layer_sizes = (2048,15))
    mlp.fit(X_train_data, y_train)
    print('Accuracy of MLP classifier on test set: {:.2f}'.format(mlp.score(X_test_data, y_test)))
    print('Accuracy of MLP classifier on training set: {:.2f}'.format(mlp.score(X_train_data, y_train)))
    scores = cross_val_score(mlp, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    linsvc = LinearSVC(C = float(c))
    linsvc.fit(X_train_data, y_train)
    print('Accuracy of LinearSVC classifier on test set: {:.2f}'.format(linsvc.score(X_test_data, y_test)))
    print('Accuracy of LinearSVC classifier on training set: {:.2f}'.format(linsvc.score(X_train_data, y_train)))
    scores = cross_val_score(linsvc, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))   
    
    svc = SVC(C = float(c), kernel = 'rbf', gamma = 'scale')
    svc.fit(X_train_data, y_train)
    print('Accuracy of SVC on test set: {:.2f}'.format(svc.score(X_test_data, y_test)))
    print('Accuracy of SVC on training set: {:.2f}'.format(svc.score(X_train_data, y_train)))
    scores = cross_val_score(svc, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))    
    
    rfc = RandomForestClassifier(n_estimators=1000, class_weight = {0:0.00000002, 1:0.99999998})
    rfc.fit(X_train_data, y_train)
    print('Accuracy of RFC on test set: {:.2f}'.format(rfc.score(X_test_data, y_test)))
    print('Accuracy of RFC on training set: {:.2f}'.format(rfc.score(X_train_data, y_train)))
    scores = cross_val_score(rfc, X_data, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 
    
    y_pred = mlp.predict(X_test_data)
    
    optimized_data = pd.DataFrame(data=X_test)
    optimized_data.insert(3,'pred', y_pred)

    if (testdata is not None):
        for_prediction = processed_test_data.iloc[:, 3:]
        test_pred = mlp.predict(for_prediction)
        processed_test_data['pred'] = test_pred
        successes = 0
        failures = 0
        correct_ids = 0
        for index, row in processed_test_data.iterrows():
            for i, r in optimized_data.iterrows():
                if r['CatalystName']==row['CatalystName']:
                    if r['ActiveSite']==row['ActiveSite']:
                        if r['Label'] == row['pred']:
                            correct_ids += 1
                            print('Correct id!')
                        if r['pred'] == row['pred']:
                            successes += 1
                            print("Success!")
                        else:
                            failures += 1
                            print("Failure")
        print("Total matches between optimized data and forcefield data: ", successes, "out of ", successes+failures)
        print("Total correct identifications: ", correct_ids, "out of ", successes+failures)

    classifier_dict = {"dtc" : dtc, "log" : logregression, "svc" : svc, "mlp" : mlp, "linsvc" : linsvc}


    if len(cols) == 2:
        plotOverpotentialClassification(X_test_data.values[:,:],y_test,mlp, rfc, linsvc, logregression)
    
    if confusion!=0:
        plot_confusion_matrix(y_test, y_pred) 
    
    if plotcoef!=0:
        plot_coefficients(linsvc, cols)



def linearRegression(alldata, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20):

    print(alldata.size)
    alldata = alldata[(alldata["Doesitbind"] == True) & (alldata["BindingEnergy"]>-1.5)]
    print(alldata.size)
        
    cols = []
    features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20]
    for feature in features:
        if feature != None:
            cols.append(feature)
    
    print("For the features ", cols)

    X=alldata[cols]
    X = X.values
    y=alldata['BindingEnergy'].astype('float')
    y = y.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    lin_model = linear_model.LinearRegression()
    svr = svm.SVR(kernel = 'rbf', gamma = 'auto')
    gpr = GaussianProcessRegressor()
    # Train the model using the training sets
    lin_model.fit(X_train, y_train)
    svr.fit(X_train, y_train)
    gpr.fit(X_train, y_train)
    # Make predictions using the testing set
    y_pred_lin = lin_model.predict(X_test)
    y_pred_svr = svr.predict(X_test)
    y_pred_gpr = gpr.predict(X_test)
    # The coefficients
    print('Coefficients: \n', lin_model.coef_)
    # The mean squared error
    print("Root mean squared error, linear: %.2f"
          % sqrt(mean_squared_error(y_test, y_pred_lin)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred_lin))
    print("Root mean squared error, svr: %.2f"
          % sqrt(mean_squared_error(y_test, y_pred_svr)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred_svr))
    print("Root mean squared error, gpr: %.2f"
          % sqrt(mean_squared_error(y_test, y_pred_gpr)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred_gpr))

    # Plot outputs
    #print(X_test, y_test)
    if len(cols)==1:
        plt.scatter(X, y,  color='black')
        plt.plot(X_test, y_pred_gpr, color='blue', linewidth=3)
        #plt.plot(X_test, y_pred_svr, color = 'red', linewidth = 3)

        #plt.xticks(())
        #plt.yticks(())
        plt.xlabel(cols[0])
        plt.ylabel("Binding Energy")

        plt.show()
    else:
        plt.scatter(y_test, y_pred_lin, color='blue')
        #plt.scatter(y_pred_svr, y_test, color = 'red')

        #plt.xticks(())
        #plt.yticks(())
        
        plt.xlabel("Binding energy (actual) (eV)")
        plt.ylabel("Binding energy (predicted) (eV)")
        
        plt.xlim(-1,0)
        plt.ylim(-1,0)

        plt.show()

