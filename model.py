import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')

import gc

import pandas as pd 
import numpy as np 

##################### Reading data ########################

train_data = pd.read_csv('../input/ sales_train.csv')
test_data = pd.read_csv('../input/test.csv')

print('train: ', train_data.shape,'test: ', test_data.shape)
print(test_data.head())

items = pd.read_csv('../input/items.csv')
item_cat = pd.read_csv('../input/item_categories.csv')
shops = pd.read_Csv('../input/shops.csv')


from sklearn.feature_extraction.text import TfidVectorizer

feature_cnt =30
tfid = TfidVectorizer(max_df = 0.6, max_features=feature_cnt, ngram_range=(1,2))
item_cat['item_category_name_len'] = item_cat['item_category_name'].apply(len)
item_cat['item_category_name_wc'] = item_cat['item_category_name'].apply(lambda x: len(str(x).split(' ')))
print(item_cat.head())

txtFeatures = pd.DataFrame(tfid.fit_transform(item_cat['item_category_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt):
	item_cat['item_category_name_tfid_' + str(i)] = txtFeatures[cols[i]]


items['item_name_len'] = item['item_name'].apply(len)
items['item_name_wc'] = item['item_name'].apply(lambda x: len(str(x).split(' ')))
txtFeatures = pd.DataFrame(tfid.fit_transform(item['item_name']).toarray())

cols = txtFeatures.columns
for i in range(feature_cnt):
	items['item_name_tfid_' + str(i)] = txtFeatures[cols[i]]


shops['shop_name_len'] = shops['shop_name'].apply(len)
shops['shop_name_wc'] = shops['shop_name'].apply(lambda x: len(str(x).split(' ')))
txtFeatures = pd.DataFrame(tfid.fit_transform(shops['shop_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt:
	shops['shop_name_tfid_' + str(i)]) =  txtFeatures[cols[i]]


train_data['date'] = pd.to_datetime(train_data['date'], format='%d.%m.%Y')

train_data['month'] = train_data['date'].dt.month
train_data['year'] = train_data['date'].dt.year


train_data = train_data.drop(['date','item_price'], axis=1)
train_data = train_data.groupby([c for c in train_data.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train_data = train_data.rename(columns= {'item_cnt_day': 'item_cnt_month'})


shop_item_monthly_mean = train_data[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns = {'item_cnt_month': 'item_cnt_month_mean'})

train_data = pd.merge(train_data, shop_item_monthly_mean, how='left', on=['shop_id','item_id'])


train_data = pd.merge(train_data, items, how='left', on='item_id')
train_data = pd.merge(train_data, item_cat, how='left', on='item_category_id')
train_data = pd.merge(train_data, shops, how='left', on='shop_id')




from sklearn.preprocessing import LabelEncoder

for col in train_data.columns:
	if train_data[col].dtypes =='object'
		lb = LabelEncoder()
		lb.fit(list(train_data[col].values) + list(test_data[col].values))
		train_data[col]= lb.fit_transform(train_data[col].values)
		test_data[col] = lb.fit_transform(test_data[col].values)

train_data['item_cnt_month'] = np.log1p(train_data['item_cnt_month'].clip(0., 20.))
cols = [c for c in train_data.columns if c not in ['item_cnt_month']]

x= train_data[cols]
y= train_data['item_cnt_month']

del item_cat , items , train_data, shop_item_monthly_mean, shops, txtFeatures

gc.collect()



from sklearn.model_selection import train_test_split
import lightgbm as lgb


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=64)

params = {
        'boosting_type': 'gbdt', 
        'objective': 'binary',
        'metric': 'l2_root',
        'max_depth': 16,  
        'num_leaves': 31, 
        'learning_rate': 0.25,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 1,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0, 
        'seed':1234,
        'min_data': 28,
        'min_hessian': 0.05 

        }


model = lgb.train(
            params,
            lgb.Dataset(X_train, y_train),
            num_boost_round=10000,
            valid_sets=[lgb.Dataset(X_test, y_test)],
            early_stopping_rounds=100,
            verbose_eval=25)

y_pred = model.predict(X_test, num_iteration=model.best_iteration)












