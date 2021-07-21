import numpy as np
import pandas as pd

#Question 1
def answer_one():

    # Your code here
    df = pd.read_csv('fraud_data.csv')

    count = len(df[df['Class'] == 1])
    out = count / len(df)

    # Return your answer
    return out


#GIVEN
# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#Question 2
def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score

    # Your code here
    dummy_maj = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    y_dummy_pred = dummy_maj.predict(X_test)

    accuracy = dummy_maj.score(X_test, y_test)
    recall = recall_score(y_test, y_dummy_pred)

    # Return your answer
    return (accuracy, recall)


#Question 3
def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    # Your code here
    svm = SVC().fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    accuracy = svm.score(X_test, y_test)
    recall = recall_score(y_test, svm_pred)
    precision = precision_score(y_test, svm_pred)

    # Return your answer
    return (accuracy, recall, precision)


#Question 4
def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    # Your code here
    gamma = 1e-07
    C = 1e9

    svm = SVC(gamma = gamma, C = C).fit(X_train, y_train)
    svm_pred = svm.decision_function(X_test) > -220

    confusion = confusion_matrix(y_test, svm_pred)

    # Return your answer
    return confusion


#Question 5
# Your code here
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc

lr = LogisticRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)

precision, recall, thresholds = precision_recall_curve(y_test, lr_pred)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

plt.figure()
plt.xlim([-0.01, 1.1])
plt.ylim([-0.01, 1.1])
plt.plot(precision, recall)
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c = 'r', mew = 3)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-Recall Curve')
plt.axes().set_aspect('equal')
plt.show()

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.xlim([-0.01, 1.1])
plt.ylim([-0.01, 1.1])
plt.plot(fpr_lr, tpr_lr, lw = 3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LogRegr ROC Curve (area = {:.2f})'.format(roc_auc_lr))
plt.axes().set_aspect('equal')
plt.show()

a = np.interp(0.75, precision, recall)
b = np.interp(0.16, fpr_lr, tpr_lr)

(a, b)

def answer_five():
    # Your code here

    # Return your answer
    return (0.8, 0.9)


#Question 6
def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    # Your code here
    grid_val = {'penalty' : ['l1', 'l2'], 'C' : [0.01, 0.1, 1, 10, 100]}

    lr = LogisticRegression().fit(X_train, y_train)

    grid_lr = GridSearchCV(lr, param_grid = grid_val, scoring = 'recall').fit(X_train, y_train)

    out = grid_lr.cv_results_['mean_test_score'].reshape(5,2)

    # Return your answer
    return out


#GIVEN
# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    %matplotlib notebook
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

GridSearch_Heatmap(answer_six())
