from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import numpy as np

def cramers_v(confusion_matrix): 
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

def baseline_model(data,features,random_state=8964):

    train_data   = data[features].drop(columns = ['diagnosed_diabetes'])
    train_target = data['diagnosed_diabetes']

    train_data, valid_data, train_target, valid_target = train_test_split(train_data, train_target, test_size=0.2, random_state=random_state)

    log_clf   = LogisticRegression(max_iter=2000)
    rf_clf    = RandomForestClassifier(n_estimators=100, max_depth=10)

    for clf in (log_clf, rf_clf):
        clf.fit(train_data, train_target)
        y_pred = clf.predict(valid_data)
        print(clf.__class__.__name__, accuracy_score(valid_target, y_pred))

    return log_clf , rf_clf


def evalulate_model_accuracy(log_clf, rf_clf, valid_data, valid_target):

    for model in (log_clf, rf_clf):
        y_pred = model.predict(valid_data)
        y_pred_proba = model.predict_proba(valid_data)[:, 1]
        print("Accuracy: ",model.__class__.__name__, accuracy_score(valid_target, y_pred))
        print("ROC-AUC : ",model.__class__.__name__, roc_auc_score(valid_target, y_pred_proba))

        print("Classification Report:")
        print(classification_report(valid_target, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(valid_target, y_pred))
