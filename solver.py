import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

np.set_printoptions(threshold=sys.maxsize)

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")




def main():
  acquire_data()
  # understand_data()
  prepare_data()
  make_train_test_set()
  train_model()
  predict()
  write_result_csv()



def acquire_data():
  global train_df, test_df
  train_df = pd.read_csv('train.csv', header=0)
  test_df = pd.read_csv('test.csv', header=0)



def understand_data():
  print(train_df.head())  
  print(train_df.info())
  print(train_df.describe())


def plot_top_corr_heatmap():
  corr_matrix = train_df.corr(method='pearson')
  cols = corr_matrix.nlargest(10, 'SalePrice').index
  largest_corr_matrix = np.corrcoef(train_df[cols].values.T)
  sns.heatmap(largest_corr_matrix, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
  plt.show()
  


def prepare_data():
  global train_df, test_df
  train_df = grasp_features_of_top_corr(train_df)
  test_df = grasp_features_of_top_corr(test_df)
  # train_df = drop_useless_features_of(train_df)
  # test_df = drop_useless_features_of(test_df)
  handle_missing_data_for(train_df)
  handle_missing_data_for(test_df)
  # train_df = one_hot_encode_categorical_features_of(train_df)
  # test_df = one_hot_encode_categorical_features_of(test_df)
  

def grasp_features_of_top_corr(dataset_df):
  selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
  if isTestset(dataset_df):
    return dataset_df[selected_features]
  else:
    return dataset_df[selected_features + ['SalePrice']]
    

def drop_useless_features_of(dataset_df):
  return dataset_df.drop(['GarageYrBlt', 'Id'], axis=1)


def handle_missing_data_for(dataset_df):
  # dataset_df['LotFrontage'].fillna(dataset_df['LotFrontage'].dropna().median(), inplace=True)
  # dataset_df['MasVnrArea'].fillna(dataset_df['MasVnrArea'].dropna().median(), inplace=True)
  # dataset_df['MasVnrType'].fillna('None', inplace=True)
  # dataset_df['Electrical'].fillna('SBrkr', inplace=True)
  if isTestset(dataset_df):
    # dataset_df['BsmtFinSF1'].fillna(dataset_df['BsmtFinSF1'].dropna().median(), inplace=True)
    # dataset_df['BsmtFinSF2'].fillna(dataset_df['BsmtFinSF2'].dropna().median(), inplace=True)
    # dataset_df['BsmtUnfSF'].fillna(dataset_df['BsmtUnfSF'].dropna().median(), inplace=True)
    # dataset_df['TotalBsmtSF'].fillna(dataset_df['TotalBsmtSF'].dropna().median(), inplace=True)
    # dataset_df['BsmtFullBath'].fillna(0, inplace=True)
    # dataset_df['BsmtHalfBath'].fillna(0, inplace=True)
    dataset_df['GarageCars'].fillna(2, inplace=True)
    dataset_df['GarageArea'].fillna(dataset_df['GarageArea'].dropna().median(), inplace=True)


def isTestset(dataset_df):
  if 'SalePrice' not in dataset_df.columns.values:
    return True
  else:
    return False


def transform_data(parameter_list):
  pass


def one_hot_encode_categorical_features_of(dataset_df):
  return pd.get_dummies(dataset_df, dummy_na=True, drop_first=True)



def make_train_test_set():
  TARGET = 'SalePrice'
  global train_df, X_train, y_train, X_test, y_test, X_train_quad, X_test_quad
  X_train = train_df.drop([TARGET], axis=1)
  y_train = train_df[TARGET]
  X_test = test_df
  # X_train, X_test, y_train, y_test = train_test_split(train_df.drop([TARGET], axis=1), train_df[TARGET], random_state=4)
  X_train_quad = X_train.copy()
  X_train_quad['OverallQual'] = X_train_quad['OverallQual'] ** 2
  X_test_quad = X_test.copy()
  X_test_quad['OverallQual'] = X_test_quad['OverallQual'] ** 2



def train_model():
  global linear_regression, linear_regression_quad
  linear_regression = LinearRegression()
  linear_regression_quad = LinearRegression()
  linear_regression.fit(X_train, y_train)
  linear_regression_quad.fit(X_train_quad, y_train)

  print(X_train)
  print(X_train_quad)

def predict():
  global y_pred, y_pred_quad
  y_pred = linear_regression.predict(X_test)
  y_pred_quad = linear_regression_quad.predict(X_test_quad)

  # y_train_pred = linear_regression.predict(X_train)

  # y_train_log = np.log(y_train)
  # y_train_pred_log = np.log(y_train_pred)
  # y_test_log = np.log(y_test)
  # y_pred_log = np.log(y_pred)
  # print('training RMSE: ', np.sqrt(mean_squared_error(y_train_pred_log, y_train_log)))
  # print('testing RMSE: ', np.sqrt(mean_squared_error(y_test_log, y_pred_log)))



def write_result_csv():
  filename = 'submission.csv'
  f = open(filename, 'w')

  headers = 'Id,SalePrice\n'
  f.write(headers)
  START_ID = train_df.shape[0] + 1
  TESTSET_SIZE = test_df.shape[0]
  END_ID = START_ID + TESTSET_SIZE
  for i in range(START_ID, END_ID):
    current_house = str(i) + ',' + str(y_pred_quad[i - START_ID]) + '\n'
    f.write(current_house)
  
  print('File writing done.')


def log_if_missing_data_exists(dataset_df):
  print('There is data missing in dataset: ', 'YES' if dataset_df.isnull().sum().max() > 0 else 'NO')



main()






# Don't know the purpose of this piece of code in prepare_data()
  # for trainset_feature in train_df.columns.values:
  #   if trainset_feature not in test_df.columns.values and trainset_feature != 'SalePrice':
  #     test_df[trainset_feature] = 0