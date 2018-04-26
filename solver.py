import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, \
                                 ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, \
                             ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb

np.set_printoptions(threshold=sys.maxsize)

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

TARGET = 'SalePrice'


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
  global train_df, test_df, target_col

  train_df = pd.read_csv('train.csv', header=0)
  test_df = pd.read_csv('test.csv', header=0)
  target_col = train_df[TARGET]


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

  drop_features_from([TARGET], train_df)
  train_df, target_col = remove_outliers_in(train_df, target_col)
  combined_df = concat_train_test_data(train_df, test_df)
  # combined_df = grasp_features_of_top_corr(combined_df)
  handle_missing_data(combined_df)
  combined_df = create_new_features(combined_df)
  combined_df = transform_features(combined_df)
  drop_features_from(['Utilities', 'MiscFeature', 'MiscVal', 'BsmtFinSF2',
                      'LowQualFinSF', 'Exterior2nd', 'PoolArea', 'PoolQC',
                      'Condition2', 'LandSlope', 'Street', 'Heating'], combined_df)
  # combined_df = one_hot_encode_categorical_features_of(combined_df)
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
    

def drop_features_from(feats_to_drop, dataset_df):
  dataset_df.drop(feats_to_drop, axis=1, inplace=True)


def handle_missing_data(dataset_df):
  for feature in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
                  "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                  "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                  "BsmtFinType2", "MasVnrType"):
    dataset_df[feature] = dataset_df[feature].fillna("None")

  dataset_df["LotFrontage"] = dataset_df.groupby("Neighborhood")["LotFrontage"].transform(
                            lambda x: x.fillna(x.median()))

  for feature in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", 
              "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea",
              "BsmtFullBath", "BsmtHalfBath"):
    dataset_df[feature] = dataset_df[feature].fillna(0)                  
    
  dataset_df['MSZoning'] = dataset_df['MSZoning']\
                           .fillna(dataset_df['MSZoning'].mode()[0])
  dataset_df['Electrical'] = dataset_df['Electrical']\
                             .fillna(dataset_df['Electrical'].mode()[0])
  dataset_df['KitchenQual'] = dataset_df['KitchenQual']\
                              .fillna(dataset_df['KitchenQual'].mode()[0])
  dataset_df['Exterior1st'] = dataset_df['Exterior1st']\
                              .fillna(dataset_df['Exterior1st'].mode()[0])
  dataset_df['Exterior2nd'] = dataset_df['Exterior2nd']\
                              .fillna(dataset_df['Exterior2nd'].mode()[0])
  dataset_df['SaleType'] = dataset_df['SaleType']\
                           .fillna(dataset_df['SaleType'].mode()[0])
  dataset_df["Functional"] = dataset_df["Functional"]\
                             .fillna(dataset_df['Functional'].mode()[0])



def create_new_features(dataset_df):
  dataset_df["OverallQual-2"] = dataset_df["OverallQual"] ** 2
  dataset_df["GrLivArea-2"] = dataset_df["GrLivArea"] ** 2
  dataset_df["GarageCars-2"] = dataset_df["GarageCars"] ** 2
  dataset_df["GarageArea-2"] = dataset_df["GarageArea"] ** 2
  dataset_df["TotalBsmtSF-2"] = dataset_df["TotalBsmtSF"] ** 2
  dataset_df["1stFlrSF-2"] = dataset_df["1stFlrSF"] ** 2
  dataset_df["FullBath-2"] = dataset_df["FullBath"] ** 2
  dataset_df["TotRmsAbvGrd-2"] = dataset_df["TotRmsAbvGrd"] ** 2
  dataset_df["Fireplaces-2"] = dataset_df["Fireplaces"] ** 2
  dataset_df["MasVnrArea-2"] = dataset_df["MasVnrArea"] ** 2
  dataset_df["BsmtFinSF1-2"] = dataset_df["BsmtFinSF1"] ** 2
  dataset_df["LotFrontage-2"] = dataset_df["LotFrontage"] ** 2
  dataset_df["WoodDeckSF-2"] = dataset_df["WoodDeckSF"] ** 2
  dataset_df["OpenPorchSF-2"] = dataset_df["OpenPorchSF"] ** 2
  dataset_df["2ndFlrSF-2"] = dataset_df["2ndFlrSF"] ** 2

  dataset_df["OverallQual-3"] = dataset_df["OverallQual"] ** 3
  dataset_df["GrLivArea-3"] = dataset_df["GrLivArea"] ** 3
  dataset_df["GarageCars-3"] = dataset_df["GarageCars"] ** 3
  dataset_df["GarageArea-3"] = dataset_df["GarageArea"] ** 3
  dataset_df["TotalBsmtSF-3"] = dataset_df["TotalBsmtSF"] ** 3
  dataset_df["1stFlrSF-3"] = dataset_df["1stFlrSF"] ** 3
  dataset_df["FullBath-3"] = dataset_df["FullBath"] ** 3
  dataset_df["TotRmsAbvGrd-3"] = dataset_df["TotRmsAbvGrd"] ** 3
  dataset_df["Fireplaces-3"] = dataset_df["Fireplaces"] ** 3
  dataset_df["MasVnrArea-3"] = dataset_df["MasVnrArea"] ** 3
  dataset_df["BsmtFinSF1-3"] = dataset_df["BsmtFinSF1"] ** 3
  dataset_df["LotFrontage-3"] = dataset_df["LotFrontage"] ** 3
  dataset_df["WoodDeckSF-3"] = dataset_df["WoodDeckSF"] ** 3
  dataset_df["OpenPorchSF-3"] = dataset_df["OpenPorchSF"] ** 3
  dataset_df["2ndFlrSF-3"] = dataset_df["2ndFlrSF"] ** 3

  dataset_df["OverallQual-Sq"] = np.sqrt(dataset_df["OverallQual"])
  dataset_df["GrLivArea-Sq"] = np.sqrt(dataset_df["GrLivArea"])
  dataset_df["GarageCars-Sq"] = np.sqrt(dataset_df["GarageCars"])
  dataset_df["GarageArea-Sq"] = np.sqrt(dataset_df["GarageArea"])
  dataset_df["TotalBsmtSF-Sq"] = np.sqrt(dataset_df["TotalBsmtSF"])
  dataset_df["1stFlrSF-Sq"] = np.sqrt(dataset_df["1stFlrSF"])
  dataset_df["FullBath-Sq"] = np.sqrt(dataset_df["FullBath"])
  dataset_df["TotRmsAbvGrd-Sq"] = np.sqrt(dataset_df["TotRmsAbvGrd"])
  dataset_df["Fireplaces-Sq"] = np.sqrt(dataset_df["Fireplaces"])
  dataset_df["MasVnrArea-Sq"] = np.sqrt(dataset_df["MasVnrArea"])
  dataset_df["BsmtFinSF1-Sq"] = np.sqrt(dataset_df["BsmtFinSF1"])
  dataset_df["LotFrontage-Sq"] = np.sqrt(dataset_df["LotFrontage"])
  dataset_df["WoodDeckSF-Sq"] = np.sqrt(dataset_df["WoodDeckSF"])
  dataset_df["OpenPorchSF-Sq"] = np.sqrt(dataset_df["OpenPorchSF"])
  dataset_df["2ndFlrSF-Sq"] = np.sqrt(dataset_df["2ndFlrSF"])
  return dataset_df


def transform_features(dataset_df):
  dataset_df['BsmtQual'] = dataset_df['BsmtQual']\
                           .map({"None":0, "Fa":1, "TA":2, "Gd":3, "Ex":4})
  dataset_df['BsmtCond'] = dataset_df['BsmtCond']\
                           .map({"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
  dataset_df['BsmtExposure'] = dataset_df['BsmtExposure']\
                               .map({"None":0, "No":1, "Mn":2, "Av":3, "Gd":4})                           

  dataset_df = pd.get_dummies(dataset_df, columns=["BsmtFinType1"], prefix="BsmtFinType1")

  dataset_df.loc[dataset_df['BsmtFinSF1']<=1002.5, 'BsmtFinSF1'] = 1
  dataset_df.loc[(dataset_df['BsmtFinSF1']>1002.5) 
               & (dataset_df['BsmtFinSF1']<=2005), 'BsmtFinSF1'] = 2
  dataset_df.loc[(dataset_df['BsmtFinSF1']>2005) 
               & (dataset_df['BsmtFinSF1']<=3007.5), 'BsmtFinSF1'] = 3
  dataset_df.loc[dataset_df['BsmtFinSF1']>3007.5, 'BsmtFinSF1'] = 4
  dataset_df['BsmtFinSF1'] = dataset_df['BsmtFinSF1'].astype(int)
  dataset_df = pd.get_dummies(dataset_df, columns=["BsmtFinSF1"], prefix="BsmtFinSF1")
  
  dataset_df = pd.get_dummies(dataset_df, columns=["BsmtFinType2"], prefix="BsmtFinType2")

  dataset_df['BsmtFinSf2_Flag'] = dataset_df['BsmtFinSF2'].map(lambda x:0 if x==0 else 1)

  dataset_df.loc[dataset_df['BsmtUnfSF']<=778.667, 'BsmtUnfSF'] = 1
  dataset_df.loc[(dataset_df['BsmtUnfSF']>778.667) 
                         & (dataset_df['BsmtUnfSF']<=1557.333), 'BsmtUnfSF'] = 2
  dataset_df.loc[dataset_df['BsmtUnfSF']>1557.333, 'BsmtUnfSF'] = 3
  dataset_df['BsmtUnfSF'] = dataset_df['BsmtUnfSF'].astype(int)
  dataset_df = pd.get_dummies(dataset_df, columns=["BsmtUnfSF"], prefix="BsmtUnfSF")  

  # TotalBsmtSF

  # 1stFlrSF

  # 2ndFlrSF

  dataset_df['LowQualFinSF_Flag'] = dataset_df['LowQualFinSF']\
                                      .map(lambda x:0 if x==0 else 1)

  # TotalBathrooms

  dataset_df['KitchenQual'] = dataset_df['KitchenQual']\
                              .map({"Fa":1, "TA":2, "Gd":3, "Ex":4})

  dataset_df['FireplaceQu'] = dataset_df['FireplaceQu']\
                              .map({"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})

  dataset_df['MSSubClass'] = dataset_df['MSSubClass'].astype(str)
  dataset_df = pd.get_dummies(dataset_df, columns=["MSSubClass"], prefix="MSSubClass")

  dataset_df['BldgType'] = dataset_df['BldgType'].astype(str)
  dataset_df = pd.get_dummies(dataset_df, columns=["BldgType"], prefix="BldgType")  

  dataset_df['HouseStyle'] = dataset_df['HouseStyle'].map({"2Story":"2Story", 
                                                      "1Story":"1Story", 
                                                      "1.5Fin":"1.5Story", 
                                                      "1.5Unf":"1.5Story", 
                                                      "SFoyer":"SFoyer", 
                                                      "SLvl":"SLvl", 
                                                      "2.5Unf":"2.5Story", 
                                                      "2.5Fin":"2.5Story"})
  dataset_df = pd.get_dummies(dataset_df, columns=["HouseStyle"], prefix="HouseStyle")

  # Remod_Diff

  dataset_df = pd.get_dummies(dataset_df, columns=["Foundation"], prefix="Foundation")

  dataset_df['Functional'] = dataset_df['Functional'].map(
                  {"Sev":1, "Maj2":2, "Maj1":3, "Mod":4, "Min2":5, "Min1":6, "Typ":7})

  dataset_df = pd.get_dummies(dataset_df, columns=["RoofStyle"], prefix="RoofStyle")

  dataset_df = pd.get_dummies(dataset_df, columns=["RoofMatl"], prefix="RoofMatl")

  def Exter2(col):
    if col['Exterior2nd'] == col['Exterior1st']:
        return 1
    else:
        return 0
  dataset_df['ExteriorMatch_Flag'] = dataset_df.apply(Exter2, axis=1)
  dataset_df = pd.get_dummies(dataset_df, columns=["Exterior1st"], prefix="Exterior1st")

  dataset_df = pd.get_dummies(dataset_df, columns=["MasVnrType"], prefix="MasVnrType")

  dataset_df['ExterQual'] = dataset_df['ExterQual'].map({"Fa":1, "TA":2, "Gd":3, "Ex":4})

  dataset_df = pd.get_dummies(dataset_df, columns=["ExterCond"], prefix="ExterCond")

  dataset_df = pd.get_dummies(dataset_df, columns=["GarageType"], prefix="GarageType")

  dataset_df = pd.get_dummies(dataset_df, columns=["GarageFinish"], prefix="GarageFinish")

  # GarageArea

  dataset_df['GarageQual'] = dataset_df['GarageQual'].map(
    {"None":0, "Po":1, "Fa":1, "TA":2, "Gd":3, "Ex":3})

  dataset_df['GarageCond'] = dataset_df['GarageCond'].map(
    {"None":"None", "Po":"Low", "Fa":"Low", "TA":"TA", "Gd":"High", "Ex":"High"})
  dataset_df = pd.get_dummies(dataset_df, columns=["GarageCond"], prefix="GarageCond")

  # WoodDeckSF

  # TotalPorchSF

  def PoolFlag(col):
    if col['PoolArea'] == 0:
        return 0
    else:
        return 1
  dataset_df['HasPool_Flag'] = dataset_df.apply(PoolFlag, axis=1)

  dataset_df = pd.get_dummies(dataset_df, columns=["Fence"], prefix="Fence")

  dataset_df = pd.get_dummies(dataset_df, columns=["MSZoning"], prefix="MSZoning")

  dataset_df = pd.get_dummies(dataset_df, columns=["Neighborhood"], prefix="Neighborhood")

  dataset_df['Condition1'] = dataset_df['Condition1'].map(
    {"Norm":"Norm", "Feedr":"Street", "PosN":"Pos", "Artery":"Street", "RRAe":"Train",
     "RRNn":"Train", "RRAn":"Train", "PosA":"Pos", "RRNe":"Train"})
  dataset_df['Condition2'] = dataset_df['Condition2'].map(
    {"Norm":"Norm", "Feedr":"Street", "PosN":"Pos", "Artery":"Street", "RRAe":"Train",
     "RRNn":"Train", "RRAn":"Train", "PosA":"Pos", "RRNe":"Train"})
  def ConditionMatch(col):
    if col['Condition1'] == col['Condition2']:
        return 0
    else:
        return 1
  dataset_df['Diff2ndCondition_Flag'] = dataset_df.apply(ConditionMatch, axis=1)
  dataset_df = pd.get_dummies(dataset_df, columns=["Condition1"], prefix="Condition1")

  dataset_df.loc[dataset_df['LotArea']<=5684.75, 'LotArea'] = 1
  dataset_df.loc[(dataset_df['LotArea']>5684.75) 
                 & (dataset_df['LotArea']<=7474), 'LotArea'] = 2
  dataset_df.loc[(dataset_df['LotArea']>7474) 
                 & (dataset_df['LotArea']<=8520), 'LotArea'] = 3
  dataset_df.loc[(dataset_df['LotArea']>8520) 
                 & (dataset_df['LotArea']<=9450), 'LotArea'] = 4
  dataset_df.loc[(dataset_df['LotArea']>9450) 
                 & (dataset_df['LotArea']<=10355.25), 'LotArea'] = 5
  dataset_df.loc[(dataset_df['LotArea']>10355.25) 
                 & (dataset_df['LotArea']<=11554.25), 'LotArea'] = 6
  dataset_df.loc[(dataset_df['LotArea']>11554.25) 
                 & (dataset_df['LotArea']<=13613), 'LotArea'] = 7
  dataset_df.loc[dataset_df['LotArea']>13613, 'LotArea'] = 8
  dataset_df['LotArea'] = dataset_df['LotArea'].astype(int)
  dataset_df = pd.get_dummies(dataset_df, columns=["LotArea"], prefix="LotArea")

  dataset_df = pd.get_dummies(dataset_df, columns=["LotShape"], prefix="LotShape")

  dataset_df = pd.get_dummies(dataset_df, columns=["LandContour"], prefix="LandContour")

  dataset_df = pd.get_dummies(dataset_df, columns=["LotConfig"], prefix="LotConfig")

  dataset_df['GentleSlope_Flag'] = dataset_df['LandSlope'].map({"Gtl":1, "Mod":0, "Sev":0})

  dataset_df = pd.get_dummies(dataset_df, columns=["Alley"], prefix="Alley")

  dataset_df = pd.get_dummies(dataset_df, columns=["PavedDrive"], prefix="PavedDrive")

  dataset_df['GasA_Flag'] = dataset_df['Heating'].map(
    {"GasA":1, "GasW":0, "Grav":0, "Wall":0, "OthW":0, "Floor":0})

  dataset_df['HeatingQC'] = dataset_df['HeatingQC'].map(
    {"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})

  dataset_df['CentralAir'] = dataset_df['CentralAir'].map({"Y":1, "N":0})

  dataset_df['Electrical'] = dataset_df['Electrical'].map(
    {"SBrkr":"SBrkr", "FuseF":"Fuse", "FuseA":"Fuse", "FuseP":"Fuse", "Mix":"Mix"})
  dataset_df = pd.get_dummies(dataset_df, columns=["Electrical"], prefix="Electrical")

  dataset_df = pd.get_dummies(dataset_df, columns=["MoSold"], prefix="MoSold")

  dataset_df = pd.get_dummies(dataset_df, columns=["YrSold"], prefix="YrSold")

  dataset_df['SaleType'] = dataset_df['SaleType'].map(
    {"WD":"WD", "New":"New", "COD":"COD", "CWD":"CWD", 
      "ConLD":"Oth", "ConLI":"Oth", "ConLw":"Oth", "Con":"Oth", "Oth":"Oth"})
  dataset_df = pd.get_dummies(dataset_df, columns=["SaleType"], prefix="SaleType")

  dataset_df = pd.get_dummies(dataset_df, columns=["SaleCondition"], prefix="SaleCondition")

  return dataset_df


def one_hot_encode_categorical_features_of(dataset_df):
  pass
  # return pd.get_dummies(dataset_df, 
  #                       columns=['Neighborhood'], prefix='Neighborhood', 
  #                       dummy_na=True, drop_first=True)
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