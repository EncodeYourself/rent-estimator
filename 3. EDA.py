#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

pd.set_option('display.max_columns', None)
#%%
df = pd.read_csv(os.getcwd() + '/processed_dataset.csv', index_col=0)
districts = df.columns[7:20]

#%% 
'''Checking the correlation. There are some columns with ordinal values, 
I decided to add the kendall's tau column.'''
pearson = df.corr('pearson')['price'][:]
kendall = df.corr('kendall')['price'][:]
corr = pd.concat([pearson, kendall], axis=1).drop('price')
corr.columns = ['pearson', 'kendall']
sorted_indices = (corr['pearson'] + corr['kendall']).sort_values(ascending = False).index
corr.loc[sorted_indices, :]

#%% 
'''I know for sure that the data is quite noisy. Not just because some failed to
correctly fill the fields, but because of the subjectivity of the prices as well'''
fig, (ax1, ax2) = plt.subplots(ncols = 2)
sns.boxenplot(x = (df['m2_size']), ax = ax1)
sns.boxenplot(x = (df['price']), ax = ax2)
df.groupby('Type').describe()[['price', 'm2_size']]

#%% 
'''Now time to see how many objects are located in the districts and outside'''
dist_counts = df.loc[:, districts].sum()
new_row = {'Outside of the city': df.shape[0] - dist_counts.sum()}
dist_counts.append(pd.Series(new_row)).sort_values(ascending=False)

#%%
'''Let's see the median rents and sizes for the districts'''
for district in districts:
    info = df.loc[df[district] == 1, 'price'].median()
    size = df.loc[df[district] == 1, 'm2_size'].median()
    print(f'{district:<50} {size :^} \t {info: >}')
#%%
'''The distance from the objects to the closest metro station'''
sns.countplot(x = df.distance)