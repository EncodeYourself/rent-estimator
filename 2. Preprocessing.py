from bs4 import BeautifulSoup as BS
import numpy as np
import os
import pandas as pd
import requests

#%%
df1 = pd.read_csv(os.getcwd()+'/database.csv', delimiter=',')
df2 = pd.read_csv(os.getcwd()+'/database2.csv', delimiter=',')
df = pd.concat([df1,df2], axis=0)
df.drop(columns= 'string', axis=1, inplace=True)
df.drop(df[df.Type.isna()].index, inplace=True)
df['Studio'] = 0
df.loc[(df.Type =='Апартаменты-студия')|(df.Type=='Квартира-студия'), 'Studio'] = 1

#%%
df['Type'] = df['Type'].map({'1-к. квартира':1, '2-к. квартира':2, '3-к. квартира':3,
                            '4-к. квартира':4, '5-к. квартира':5, '6-к. квартира':6,
                            '7-к. квартира':7, '8-к. квартира':8, '9-к. квартира':9,
                            '10 и более-к. квартира':10, '1-к. апартаменты':1,
                            '2-к. апартаменты':2, '3-к. апартаменты':3, '4-к. апартаменты':4,
                            'Апартаменты-студия':1, 'Квартира-студия':1})
df['distance'] = df['distance'].map({'31':5, '21–30':4, '6–10':1, '5':0, '40':6, '11–15':2, '16–20':3, np.nan:6})
df.loc[df.metro.isna(), 'district'] = 'Вне города'
#%%
stations = df.district.unique()
for station1 in stations:
    for station2 in stations:
        try:
            if station1[:-1] == station2:
                df.loc[df.district==f"{station2}", 'district'] = f"{station1}"
        except TypeError:
            continue

#%%
df.drop(df[df.Type.isna()].index, inplace = True)
#%%
def metro_to_district():
    district_wikipage = r'https://ru.wikipedia.org/wiki/%D0%A1%D0%BF%D0%B8%D1%81%D0%BE%D0%BA_%D1%81%D1%82%D0%B0%D0%BD%D1%86%D0%B8%D0%B9_%D0%9F%D0%B5%D1%82%D0%B5%D1%80%D0%B1%D1%83%D1%80%D0%B3%D1%81%D0%BA%D0%BE%D0%B3%D0%BE_%D0%BC%D0%B5%D1%82%D1%80%D0%BE%D0%BF%D0%BE%D0%BB%D0%B8%D1%82%D0%B5%D0%BD%D0%B0'
    page = requests.get(district_wikipage)
    
    table = BS(page.content, 'lxml').find('table').find_all('tr')
    district_dict = {}
    for row in table:
        items = row.find_all('td')
        if len(items) > 0:
            key = items[1].text
            value = items[-2].text
            district_dict[key] = value

    for key, value in district_dict.items():
        district_dict[key] = value[:-1]
        
    return district_dict

district_dict = metro_to_district()
#%%
def apply_metro_to_dist(dataframe, dictionary):
    for station in dataframe.metro.unique():
        try:
            words = station.split(' ')
        except AttributeError:
            continue
        
        if words[0] in ['Площадь','Проспект']:
            try:
                search_word = words[2]
            except IndexError:
                search_word = words[1]
        else:
            search_word = words[0]
        
        for metro, district in dictionary.items():
            if search_word in metro:
                dataframe.loc[dataframe.metro == station, 'district'] = district.replace('\n', '')
                break
            
        return dataframe

#%%
processed_df = apply_metro_to_dist(df, district_dict)

processed_df = pd.get_dummies(df, columns = ['district', 'applicant']).drop(
    ['district_Вне города','applicant_Агентство', 'metro'], axis=1)

#%%
processed_df.to_csv('processed_dataset.csv')