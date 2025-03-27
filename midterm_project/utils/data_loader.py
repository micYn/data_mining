import numpy as np
import pandas as pd

def load_dataset(train_path, test_path):
    ''' 
    Loads a dataset from csv files 
    Input: training, testing data csv file path 
    Output: X_train: training feature
            y_train: training labels
            X_test:  testing feature
            y_test:  testing labels
    '''
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.iloc[:, :-1].values  # All columns except the last one
    y_train = train_df.iloc[:, -1].values   # the last column

    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    return X_train, y_train, X_test, y_test
