import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

def blight_model():
    # Your code here
    #Read in files
    train_df = pd.read_csv('train.csv', encoding = "ISO-8859-1")
    test_df = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    addresses_df =  pd.read_csv('addresses.csv', encoding = "ISO-8859-1")
    latlons_df = pd.read_csv('latlons.csv', encoding = "ISO-8859-1")

    #formatting indices
    train_df.set_index('ticket_id', inplace = True)
    test_df.set_index('ticket_id', inplace = True)
    latlons_df.set_index('address', inplace = True)

    #join
    addresses_df.set_index('address', inplace = True)
    addresses_df = addresses_df.join(latlons_df)

    addresses_df.set_index('ticket_id', inplace = True)
    train_df = train_df.join(addresses_df)
    test_df = test_df.join(addresses_df)

    #clean
    train_only = ['payment_amount', 'payment_date',
                      'payment_status', 'balance_due',
                      'collection_status', 'compliance_detail']
    string_cols = ['agency_name', 'inspector_name',
                   'violator_name', 'violation_street_number',
                   'violation_street_name', 'violation_zip_code',
                   'mailing_address_str_number', 'mailing_address_str_name',
                   'city', 'state', 'zip_code', 'non_us_str_code', 'country',
                   'ticket_issued_date', 'hearing_date',
                   'violation_code', 'violation_description',
                   'disposition', 'grafitti_status']


    train_df.drop(train_only, axis = 'columns', inplace = True)
    train_df.drop(string_cols, axis = 'columns', inplace = True)
    test_df.drop(string_cols, axis = 'columns', inplace = True)

    train_df['lat'].fillna(method = 'ffill', inplace = True)
    train_df['lon'].fillna(method = 'ffill', inplace = True)
    test_df['lat'].fillna(method = 'ffill', inplace = True)
    test_df['lon'].fillna(method = 'ffill', inplace = True)

    #set up train_test
    train_df = train_df[(train_df['compliance'] == 0) | (train_df['compliance'] == 1)]

    y_train = train_df['compliance']
    X_train = train_df.drop(['compliance'], axis = 'columns')
    X_test = test_df

    #classifier
    clf = MLPClassifier(hidden_layer_sizes = [10, 10], alpha = 0.01, solver = 'lbfgs', random_state = 0).fit(X_train, y_train)

    #find best metrics
    #param = {'alpha': [0.01, 0.1, 1, 10], 'hidden_layer_sizes': [[10, 10], [50, 10], [100, 10], [150, 10]]}

    #grid_clf = GridSearchCV(clf, param_grid = param, scoring = 'roc_auc').fit(X_train, y_train)

    #print('Best parameter: ', grid_clf.best_params_)
    #print('Best score: ', grid_clf.best_score_)

    #probability
    y_prob = clf.predict_proba(X_test)[:, 1]

    out = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    out.set_index('ticket_id', inplace = True)
    out['compliance'] = y_prob

    # Your answer here
    return out['compliance']
