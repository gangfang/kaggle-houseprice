import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
  acquire_data()
  # understand_data()
  prepare_data()



def acquire_data():
  global train_df, test_df
  train_df = pd.read_csv('train.csv', header=0)
  test_df = pd.read_csv('test.csv', header=0)


def understand_data():
  log_if_missing_data_exists()
  print('\n','--'*30,'\nHead: \n',train_df.head())  
  print('\n','--'*30,'\nInfo: \n')
  train_df.info()
  print('\n','--'*30,'\nDescribe: \n',train_df.describe())
  

def log_if_missing_data_exists():
  print('There is data missing in dataset: ', 'YES' if train_df.isnull().sum().max() > 0 else 'NO')


def prepare_data():
  drop_useless_features()
  handle_missing_data()
  one_hot_encode_categorical_features()

def drop_useless_features():
  global train_df
  train_df = train_df.drop(['GarageYrBlt'], axis=1)

def handle_missing_data():
  train_df['LotFrontage'].fillna(train_df['LotFrontage'].dropna().median(), inplace=True)
  train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].dropna().median(), inplace=True)
  train_df['MasVnrType'].fillna('None', inplace=True)
  train_df['Electrical'].fillna('SBrkr', inplace=True)

def one_hot_encode_categorical_features():
  global train_df
  train_df = pd.get_dummies(train_df, dummy_na=True, drop_first=True)







def check_data():
  acquire_data()





# check_data()
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