#%%
import time
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd
from scipy import stats
import seaborn as sns
import xgboost as xgb
#%%
df = pd.read_csv(getcwd() + '/processed_dataset.csv', index_col=0)

#%%
def z_score(x, y, z_score = 3, columns = ['price','m2_size'], price_limit = None):
    indexes = []
    
    if 'm2_size' in columns:
        x_indexes = x[np.abs(stats.zscore(x.loc[:, 'm2_size'])) < z_score].index
        indexes.append(x_indexes)
        
    if 'price' in columns:
        if price_limit is None:
            y_indexes = y[np.abs(stats.zscore(y.loc[:, 'price'])) < z_score].index
        else:
            y_indexes = y[(np.abs(stats.zscore(y.loc[:, 'price']) < z_score)) & (y['price'] < price_limit)].index
        indexes.append(y_indexes)
    
    if len(indexes) == 2:
        indx = indexes[0].intersection(indexes[1])
        return x.iloc[indx], y.iloc[indx]
    else:
        return x.iloc[indexes[0]], y.iloc[indexes[0]]
    
    

#%%
class dataset_processor(BaseEstimator, TransformerMixin):
    def __init__(self, types = True, applicant = True, districts = True, 
                 floors = True,  studio = True, log_x = True, 
                 normalization = True, pca = True):
         self.applicant = applicant
         self.types = types
         self.districts = districts
         self.floors = floors
         self.studio = studio
         self.normalization = normalization
         self.pca = pca
         self.log_x = log_x
         
    def fit(self, dataframe, y = None):
        self.dataframe = dataframe.copy()
        
        if self.normalization == True:
            self.max = self.dataframe['m2_size'].max()
            self.min = self.dataframe['m2_size'].min()
            
        if self.types == True and self.pca == True:
            self.pca_obj = PCA(n_components=1)
            self.dataframe['size_pca'] = self.pca_obj.fit(self.dataframe[['Type','m2_size']])
        return self
    
    def transform(self, dataframe, y = None):
        self.dataframe = dataframe.copy()

        if self.types == False:
            self.dataframe.drop('Type', axis = 1, inplace=True) 
        
        if self.applicant == False:
            self.dataframe.drop('applicant_Собственник', axis=1, inplace=True)
            
        if self.districts == False:
            for column in self.dataframe.columns:
                if 'district' in column:
                    self.dataframe.drop(column, axis=1, inplace=True)
        
        if self.floors == False:
            self.dataframe.drop(['floor', 'maxfloor'], axis=1, inplace=True)
        
        if self.studio == False:
            self.dataframe.drop('Studio', axis=1, inplace=True)
        
        if self.log_x == True:
            self.dataframe['m2_size'] = np.log2(self.dataframe['m2_size'])
        
        if self.normalization == True:
            self.dataframe['m2_size'] = (self.dataframe['m2_size']-self.min) / \
               (self.max - self.min)
            
        if self.types == True and self.pca == True:
            self.dataframe['size_pca'] = self.pca_obj.transform(self.dataframe[['Type','m2_size']])
            self.dataframe.drop(['Type','m2_size'], axis=1, inplace=True)
        
        return self.dataframe
        
#%%
params_grid1 = dict(processor__normalization = [True, False], processor__types = [True, False],
                   processor__districts = [True, False], processor__floors =[True, False],
                   processor__studio = [True, False], processor__log_x = [True, False],
                   processor__pca = [True, False], processor__applicant = [True, False]) 

params_grid2 = dict(regression__min_child_weight = [1,5,10], regression__gamma = [.5, 1, 1.5, 2, 5],
                   regression__subsample = [.6, .8, 1], regression__colsample_bytree = [.6, .8, 1],
                   regression__max_depth = [3,4,5], regression__learning_rate=[.02, .05, .1, .01],
                   regression__n_estimators=[100,300,500])

params_grid3 = dict(processor__normalization = [True, False], processor__types = [True, False],
                   processor__districts = [True, False], processor__floors =[True, False],
                   processor__studio = [True, False], processor__log_x = [True, False],
                   processor__pca = [True, False], processor__applicant = [True, False], 
                   regression__min_child_weight = [1,5,10], regression__gamma = [.5, 1, 1.5, 2, 5],
                   regression__subsample = [.6, .8, 1], regression__colsample_bytree = [.6, .8, 1],
                   regression__max_depth = [3,4,5], regression__learning_rate=[.02, .05, .1, .01],
                   regression__n_estimators=[100,300,500])

#%%
'''The GridSearch part, better to skip it since I rewrote it many times and
spent quite a while trying different settings'''
figures = []
for limit in range(30000, 60000, 10000):
    x = df.loc[:, df.columns !='price']
    y = df.loc[:, df.columns =='price']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1, shuffle=True)
    pipe = Pipeline([('processor', dataset_processor()), ('regression', xgb.XGBRegressor())])
    x_train, y_train = z_score(x_train, y_train, price_limit = limit)
    
    y_train = np.log10(y_train)
    y_test = np.log10(y_test)
        
    grid_search = GridSearchCV(pipe, param_grid = params_grid1, verbose=10)
    grid_search.fit(x_train, y_train)
    y_pred = grid_search.predict(x_test)
    score = mean_squared_error(y_test, y_pred)
    
    print(limit, score, grid_search.best_params_, grid_search.best_score_)
    
    f = plt.figure(figsize=(6,6))
    figures += [f]  
    ax = plt.axes()
    ax.set_title(f'{limit}')
    ax.plot(y_pred, y_test, marker = 'o', ls='', ms = 3.0)

#%%
'''The final pipeline that I decided to go with'''
x = df.loc[:, df.columns !='price']
y = df.loc[:, df.columns =='price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1, shuffle=True)

pipe = Pipeline([('processor', dataset_processor(applicant = True, districts = True, 
    floors = True, log_x = False, normalization = True, pca = True, studio = True, 
    types = True)), 
    ('regression', xgb.XGBRegressor(colsample_bytree = 0.8, gamma = 0.5,
    learning_rate = 0.1, max_depth = 5, min_child_weight = 5, n_estimators = 500,
    subsample = 0.8))])

x_train, y_train = z_score(x_train, y_train, price_limit=30000)

y_train = np.log10(y_train)
y_test = np.log10(y_test)

pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

score = mean_squared_error(y_test, y_pred)
print(round(score, 5))
    
#%%
'''Let's see how wrong my model is...'''
sq_y_test = y_test.squeeze()
diff = 10**sq_y_test - 10**y_pred
perc_diff = diff / 10**sq_y_test
fig, (ax1, ax2) = plt.subplots(ncols = 2)
sns.scatterplot(sq_y_test, sq_y_test - y_pred, ax = ax1, )
sns.histplot(perc_diff, ax = ax2, kde = True)
'''Could be much better'''
first_q, second_q = np.quantile(perc_diff, (.25, .75))

#%%
Object_of_Interest = [[1, 44, 16, 20, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]]
OoI = pd.DataFrame(Object_of_Interest, columns=x_train.columns)

#%%
final_result = 10**pipe.predict(OoI)[0]
lower_bound = final_result + final_result * first_q
upper_bound = final_result + final_result * second_q
print(
      f'''The final goal of this project is to determine a reasonable rent level
for the object of interest. The most probable answer, considering the model and
how bad it is, is the range of {lower_bound} - {upper_bound}.''')
