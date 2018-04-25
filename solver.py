import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# import xgboost as xgb
from sklearn.model_selection import cross_validate

np.set_printoptions(threshold=sys.maxsize)

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")




def main():
  acquire_data()
  # understand_data()
  prepare_data()
  do_cross_validation()
  train_model()
  predict()
  exponentiate_pred_result()
  write_result_csv()



def acquire_data():
  TARGET = 'SalePrice'
  global train_df, test_df, target_col

  train_df = pd.read_csv('train.csv', header=0)
  test_df = pd.read_csv('test.csv', header=0)
  target_col = train_df[TARGET]
  train_df = train_df.drop([TARGET], axis=1)


def understand_data():
  train_df.info()
  test_df.info()


def plot_top_corr_heatmap():
  corr_matrix = train_df.corr(method='pearson')
  cols = corr_matrix.nlargest(10, 'SalePrice').index
  largest_corr_matrix = np.corrcoef(train_df[cols].values.T)
  sns.heatmap(largest_corr_matrix, cbar=True, annot=True, 
              square=True, fmt='.2f', annot_kws={'size': 10}, 
              yticklabels=cols.values, xticklabels=cols.values)
  plt.show()
  


def prepare_data():
  global train_df, test_df, combined_df, target_col
  train_df, target_col = remove_outliers_in(train_df, target_col)
  combined_df = concat_train_test_data(train_df, test_df)
  combined_df = grasp_features_of_top_corr(combined_df)
  # combined_df = drop_useless_features_from(combined_df)
  handle_missing_data_for(combined_df)
  combined_df = transform_features_for(combined_df)
  combined_df = create_new_features_for(combined_df)
  combined_df = one_hot_encode_categorical_features_of(combined_df)
  split_train_test_data()
  log_transform(target_col)
  

def remove_outliers_in(train_df, target_col):
  outliers_idx = train_df[(train_df['GrLivArea']>4000)].index
  train_df = train_df.drop(outliers_idx)
  target_col = target_col.drop(outliers_idx)
  return (train_df, target_col)


def concat_train_test_data(train_df, test_df):
  return pd.concat([train_df, test_df])


def grasp_features_of_top_corr(dataset_df):
  selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', \
                       'GarageArea', 'TotalBsmtSF', 'LotFrontage',\
                       'FullBath', 'TotRmsAbvGrd', 'YearBuilt', \
                       'YearRemodAdd', 'ExterQual', 'BsmtQual', \
                       'Neighborhood', '1stFlrSF']
  return dataset_df[selected_features]
    

def drop_useless_features_from(dataset_df):
  return dataset_df.drop([], axis=1)


def handle_missing_data_for(dataset_df):
  dataset_df['LotFrontage'].fillna(dataset_df['LotFrontage'].dropna().median(), inplace=True)
  # dataset_df['MasVnrArea'].fillna(dataset_df['MasVnrArea'].dropna().median(), inplace=True)
  # dataset_df['MasVnrType'].fillna('None', inplace=True)
  dataset_df['BsmtQual'].fillna('None', inplace=True)
  # dataset_df['Electrical'].fillna('SBrkr', inplace=True)
  # dataset_df['BsmtFinSF1'].fillna(dataset_df['BsmtFinSF1'].dropna().median(), inplace=True)
  # dataset_df['BsmtFinSF2'].fillna(dataset_df['BsmtFinSF2'].dropna().median(), inplace=True)
  # dataset_df['BsmtUnfSF'].fillna(dataset_df['BsmtUnfSF'].dropna().median(), inplace=True)
  dataset_df['TotalBsmtSF'].fillna(dataset_df['TotalBsmtSF'].dropna().median(), inplace=True)
  # dataset_df['BsmtFullBath'].fillna(0, inplace=True)
  # dataset_df['BsmtHalfBath'].fillna(0, inplace=True)
  dataset_df['GarageCars'].fillna(2, inplace=True)
  dataset_df['GarageArea'].fillna(dataset_df['GarageArea'].dropna().median(), inplace=True)


def transform_features_for(dataset_df):
  QUALITY_MAP = {"None":0, "Fa":1, "TA":2, "Gd":3, "Ex":4}
  dataset_df['ExterQual'] = dataset_df['ExterQual'].map(QUALITY_MAP)
  dataset_df['BsmtQual'] = dataset_df['BsmtQual'].map(QUALITY_MAP)
  return dataset_df


def create_new_features_for(dataset_df):
  dataset_df['OverallQual_quad'] = dataset_df['OverallQual'] ** 2
  dataset_df["OverallQual_cub"] = dataset_df["OverallQual"] ** 3
  dataset_df['GrLivArea_quad'] = dataset_df['GrLivArea'] ** 2
  dataset_df['GrLivArea_cub'] = dataset_df['GrLivArea'] ** 3
  dataset_df['GarageCars_quad'] = dataset_df['GarageCars'] ** 2
  dataset_df['GarageCars_cub'] = dataset_df['GarageCars'] ** 3
  dataset_df['TotRmsAbvGrd_quad'] = dataset_df['TotRmsAbvGrd'] ** 2
  dataset_df['BsmtQual_quad'] = dataset_df['BsmtQual'] ** 2
  dataset_df['BsmtQual_cub'] = dataset_df['BsmtQual'] ** 3
  dataset_df['LotFrontage_quad'] = dataset_df['LotFrontage'] ** 2
  dataset_df['LotFrontage_cub'] = dataset_df['LotFrontage'] ** 3
  return dataset_df


def one_hot_encode_categorical_features_of(dataset_df):
  return pd.get_dummies(dataset_df, 
                        columns=['Neighborhood'], prefix='Neighborhood', 
                        dummy_na=True, drop_first=True)
  # return pd.get_dummies(dataset_df, dummy_na=True, drop_first=True)


def split_train_test_data():
  trainset_length = get_trainset_length(train_df)

  global X_train, X_pred
  X_train = combined_df.iloc[:trainset_length, :]
  X_pred = combined_df.iloc[trainset_length:, :]


def get_trainset_length(train_df):
  return train_df.shape[0]

  
def log_transform(target_col):
  global y_train 
  y_train = np.log1p(target_col)



def do_cross_validation():
  linear_regression = LinearRegression()
  scores = cross_validate(linear_regression, 
                           X_train, y_train, 
                           cv=10, return_train_score=True,
                           scoring='neg_mean_squared_error')
  train_RMSE = np.sqrt(-1 * scores['train_score']).mean()
  test_RMSE = np.sqrt(-1 * scores['test_score']).mean()
  print('train_RMSE: ', train_RMSE)
  print('test_RMSE: ', test_RMSE)



def train_model():
  global linear_regression
  linear_regression = LinearRegression()
  linear_regression.fit(X_train, y_train)



def predict():
  global y_pred
  y_pred = linear_regression.predict(X_pred)
  


def exponentiate_pred_result():
  global y_pred
  y_pred = np.exp(y_pred)



def write_result_csv():
  filename = 'submission.csv'
  START_ID = 1461
  TESTSET_SIZE = test_df.shape[0]
  END_ID = START_ID + TESTSET_SIZE
  headers = 'Id,SalePrice\n'

  f = open(filename, 'w')
  f.write(headers)
  for i in range(START_ID, END_ID):
    current_house = str(i) + ',' + str(y_pred[i - START_ID]) + '\n'
    f.write(current_house)
  
  print('File writing done.')








main()