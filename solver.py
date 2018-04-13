import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


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
  print('\n','--'*30,'\nHead: \n',train_df.head())  
  print('\n','--'*30,'\nInfo: \n')
  train_df.info()
  print('\n','--'*30,'\nDescribe: \n',train_df.describe())

  

def prepare_data():
  global train_df, test_df
  train_df = drop_useless_features_of(train_df)
  test_df = drop_useless_features_of(test_df)
  handle_missing_data_for(train_df)
  handle_missing_data_for(test_df)
  train_df = one_hot_encode_categorical_features_of(train_df)
  test_df = one_hot_encode_categorical_features_of(test_df)

  for trainset_feature in train_df.columns.values:
    if trainset_feature not in test_df.columns.values and trainset_feature != 'SalePrice':
      test_df[trainset_feature] = 0
      

def drop_useless_features_of(dataset_df):
  return dataset_df.drop(['GarageYrBlt', 'Id'], axis=1)

def handle_missing_data_for(dataset_df):
  dataset_df['LotFrontage'].fillna(dataset_df['LotFrontage'].dropna().median(), inplace=True)
  dataset_df['MasVnrArea'].fillna(dataset_df['MasVnrArea'].dropna().median(), inplace=True)
  dataset_df['MasVnrType'].fillna('None', inplace=True)
  dataset_df['Electrical'].fillna('SBrkr', inplace=True)
  if 'SalePrice' not in dataset_df.columns.values:
    dataset_df['BsmtFinSF1'].fillna(dataset_df['BsmtFinSF1'].dropna().median(), inplace=True)
    dataset_df['BsmtFinSF2'].fillna(dataset_df['BsmtFinSF2'].dropna().median(), inplace=True)
    dataset_df['BsmtUnfSF'].fillna(dataset_df['BsmtUnfSF'].dropna().median(), inplace=True)
    dataset_df['TotalBsmtSF'].fillna(dataset_df['TotalBsmtSF'].dropna().median(), inplace=True)
    dataset_df['BsmtFullBath'].fillna(0, inplace=True)
    dataset_df['BsmtHalfBath'].fillna(0, inplace=True)
    dataset_df['GarageCars'].fillna(2, inplace=True)
    dataset_df['GarageArea'].fillna(dataset_df['GarageArea'].dropna().median(), inplace=True)

def one_hot_encode_categorical_features_of(dataset_df):
  return pd.get_dummies(dataset_df, dummy_na=True, drop_first=True)


def make_train_test_set():
  TARGET = 'SalePrice'
  global train_df, X_train, y_train, X_test
  y_train = train_df[TARGET]
  X_train = train_df.drop([TARGET], axis=1)
  X_test = test_df


def train_model():
  global linear_regression
  linear_regression = LinearRegression()
  linear_regression.fit(X_train, y_train)


def predict():
  global prediction_result
  prediction_result = linear_regression.predict(X_test)


def write_result_csv():
  filename = 'submission.csv'
  f = open(filename, 'w')

  headers = 'Id,SalePrice\n'
  f.write(headers)
  START_ID = train_df.shape[0] + 1
  TESTSET_SIZE = test_df.shape[0]
  END_ID = START_ID + TESTSET_SIZE
  for i in range(START_ID, END_ID):
    current_house = str(i) + ',' + str(prediction_result[i - START_ID]) + '\n'
    f.write(current_house)


def log_if_missing_data_exists(dataset_df):
  print('There is data missing in dataset: ', 'YES' if dataset_df.isnull().sum().max() > 0 else 'NO')




main()














def make_boxplot_neighborhood_saleprice():
  matrix_Neighborhood_SalePrice = pd.concat([train_df['Neighborhood'], train_df['SalePrice']], axis=1)
  sns.boxplot(x='Neighborhood', y="SalePrice", data=matrix_Neighborhood_SalePrice)
  plt.show()
def make_swarmboxplot_bldgtype_saleprice():
  matrix_BldgType_SalePrice = pd.concat([train_df['BldgType'], train_df['SalePrice']], axis=1)
  sns.boxplot(x='BldgType', y="SalePrice", data=matrix_BldgType_SalePrice)
  sns.swarmplot(x='BldgType', y="SalePrice", data=matrix_BldgType_SalePrice)
  plt.show()
def make_violinplot_overallqual_saleprice():
  matrix_OverallQual_SalePrice = pd.concat([train_df['OverallQual'], train_df['SalePrice']], axis=1)
  sns.violinplot(x='OverallQual', y="SalePrice", data=matrix_OverallQual_SalePrice, scale='count')
  plt.show()
def make_scatterplot_grlivarea_saleprice():
  cor_GrLivArea_SalePrice = pd.concat([train_df['GrLivArea'], train_df['SalePrice']], axis=1)
  cor_GrLivArea_SalePrice.plot.scatter(x='GrLivArea', y='SalePrice')
  plt.show()
def plot_saleprice_distribution():
  sns.distplot(train_df['SalePrice'])
  print("Skewness: %f" % train_df['SalePrice'].skew())
  print("Kurtosis: %f" % train_df['SalePrice'].kurt())  
  plt.show()
def plot_features_corr_heatmap():
  corr_matrix = train_df.corr(method='pearson')
  plt.subplots(figsize=(12, 9))
  sns.heatmap(corr_matrix, vmax=.8, square=True)
  plt.show()
def plot_top_corr_heatmap():
  corr_matrix = train_df.corr(method='pearson')
  cols = corr_matrix.nlargest(30, 'SalePrice').index
  largest_corr_matrix = np.corrcoef(train_df[cols].values.T)
  sns.heatmap(largest_corr_matrix, cbar=True, annot=True, square=True, \
              fmt='.2f', annot_kws={'size': 30}, yticklabels=cols.values, xticklabels=cols.values)
  plt.show()
def make_features_pairplot():
  cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
  sns.pairplot(train_df[cols])
  plt.show() 