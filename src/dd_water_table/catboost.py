from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from dd_water_table.FeatureEngg import read_Data
import pandas as pd
import numpy as np
import os
import datetime
import seaborn as sns
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.metrics import classification_report

np.random.seed(2020)

model_path = '/home/divyansh/DataScience/kaggle_practice/Drivendata_competitions/Pump_it_up_water_table/model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
DATA_DIR = '/home/divyansh/DataScience/kaggle_practice/Drivendata_competitions/Pump_it_up_water_table/Data'
dlist=read_Data(DATA_DIR)
raw_target = dlist[0]
raw_data = dlist[1]
del dlist
raw_data['month']=pd.to_datetime(raw_data['date_recorded']).dt.month # Adding month in feature list
# Dates are converting to days since date
raw_data['date_recorded'] = pd.to_datetime(raw_data['date_recorded']).apply(
        lambda x: (datetime.datetime.today() - x).days)
recode={'functional':0,'functional needs repair':1,'non functional':2}
raw_target['status_group']=raw_target['status_group'].map(recode)
raw_data=raw_data.iloc[:,1:]
# All categorical features get nan replaces by string for catboost
cat_features=[i for i,c in enumerate(raw_data.columns) if raw_data[c].dtype=='object']
raw_data.iloc[:,cat_features] = raw_data.iloc[:,cat_features].replace('unknown', 'nan')
raw_data.iloc[:,cat_features] = raw_data.iloc[:,cat_features].replace(np.nan,'nan')

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2020)
params = {'depth':[3,6,8,10],
          'iterations':[100,1000],
          'learning_rate':[0.1,0.2],
          'l2_leaf_reg':[3,1,5,10,50],
          'border_count':[32,5,10,20,50,100,200],
          'ctr_border_count':[100,200,255],
          'loss_function':['MultiClass'],
          'task_type':["GPU"],
            'devices':['0:1'],
            'eval_metric':['Accuracy'],
            'boosting_type':["Ordered"],
            'bagging_temperature':[0,0.2],
            'use_best_model':[True],
            #'one_hot_max_size':[255]
          }
model = CatBoostClassifier(cat_features=cat_features)

print("Performing 10-fold CV on train data..")
time0=datetime.datetime.now()
grid_search = GridSearchCV(model, param_grid=params, scoring='roc_auc', n_jobs=-1, cv=cv, verbose=3)
grid_search.fit(raw_data, raw_target['status_group'])
time1=datetime.datetime.now()
print(f'Time taken for model fitting: {str(time1-time0)}')

print('Raw AUC score:', grid_search.best_score_)
for param_name in sorted(grid_search.best_params_.keys()):
    print("%s: %r" % (param_name, grid_search.best_params_[param_name]))

preds = grid_search.predict_proba(raw_data)
best_preds = [np.argmax(line) for line in preds]


print("\nValidation results: \n",classification_report(raw_target['status_group'],best_preds))

from sklearn.externals import joblib

joblib.dump(grid_search, os.path.join(model_path,'catboost_bst_model.pkl'), compress=True)
# bst = joblib.load('bst_model.pkl') # load it later