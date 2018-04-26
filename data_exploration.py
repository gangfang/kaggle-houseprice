import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats.stats import pearsonr


# TODO: think of a way to refactor this function, which is a duplication
# of acquire_data() function in solver.py
def acquire_data():
  TARGET = 'SalePrice'
  global train_df, test_df, target_col

  train_df = pd.read_csv('train.csv', header=0)
  test_df = pd.read_csv('test.csv', header=0)
  target_col = train_df[TARGET]
  train_df = train_df.drop([TARGET], axis=1)


def understand_data():
  grid = plt.GridSpec(2, 2)
  plt.subplots(figsize =(30, 15))
  plt.subplot(grid[0, 0])
  g1 = sns.regplot(x=train_df['OpenPorchSF'], y=target_col, 
      fit_reg=True, 
      label = "corr: %2f, p-value: %2f"%(pearsonr(train_df['OpenPorchSF'], target_col)))
  g1 = g1.legend(loc="best")

  plt.subplot(grid[0, 1])
  g2 = sns.regplot(x=train_df['EnclosedPorch'], y=target_col, 
      fit_reg=True, 
      label = "corr: %2f, p-value: %2f"%(pearsonr(train_df['EnclosedPorch'], target_col)))
  g2 = g2.legend(loc="best")                  
  
  plt.subplot(grid[1,0])
  g3 = sns.regplot(x=train_df['3SsnPorch'], y=target_col, 
      fit_reg=True, 
      label = "corr: %2f, p-value: %2f"%(pearsonr(train_df['3SsnPorch'], target_col)))
  g3 = g3.legend(loc="best")                  
  
  plt.subplot(grid[1,1])
  g3 = sns.regplot(x=train_df['ScreenPorch'], y=target_col, 
      fit_reg=True, 
      label = "corr: %2f, p-value: %2f"%(pearsonr(train_df['ScreenPorch'], target_col)))
  g3 = g3.legend(loc="best")                  
  
  
  
  plt.show()  





acquire_data()
understand_data()