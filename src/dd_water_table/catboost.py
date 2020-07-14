from catboost import CatBoostClassifier
from dd_water_table.FeatureEngg import read_Data
import pandas as pd
import numpy as np
import os
import datetime
import seaborn as sns
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.metrics import classification_report
import joblib

np.random.seed(2020)

# Train data formatting
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

# Setting up and training in gridsearchcv
cv = StratifiedKFold(n_splits=10, shuffle=True)
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

params = {'depth':[6,10],
          'iterations':[1000],
          'learning_rate':[0.1],
          'l2_leaf_reg':[3,10,50],
          'border_count':[254],
          'loss_function':['MultiClass'],
          'task_type':["GPU"],
            'devices':['0:1'],
            'eval_metric':['Accuracy'],
            'boosting_type':["Plain"],
            'bagging_temperature':[0.2],
            #'one_hot_max_size':[255]
          }
model = CatBoostClassifier(cat_features=cat_features)
print("Performing 10-fold CV on train data..")
time0=datetime.datetime.now()
grid_search = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs=-1, cv=cv, verbose=3)
grid_search.fit(raw_data, raw_target['status_group'])
time1=datetime.datetime.now()
print(f'Time taken for model fitting: {str(time1-time0)}')
print('Raw accuracy score:', grid_search.best_score_)
for param_name in sorted(grid_search.best_params_.keys()):
    print("%s: %r" % (param_name, grid_search.best_params_[param_name]))
preds = grid_search.predict_proba(raw_data)
best_preds = [np.argmax(line) for line in preds]
print("\nValidation results: \n",classification_report(raw_target['status_group'],best_preds))
from sklearn.externals import joblib
joblib.dump(grid_search, os.path.join(model_path,'catboost_bst_model.pkl'), compress=True)
print("Catboost model trained and saved!")

# Making test set predictions
test1=pd.read_csv('test.csv')
test=test1.iloc[:,1:]
test['month']=pd.to_datetime(test['date_recorded']).dt.month # Adding month in feature list
# Dates are converting to days since date
test['date_recorded'] = pd.to_datetime(test['date_recorded']).apply(
        lambda x: (datetime.datetime.today() - x).days)
cat_features=[i for i,c in enumerate(test.columns) if test[c].dtype=='object']
test.iloc[:,cat_features] = test.iloc[:,cat_features].replace('unknown', 'nan')
test.iloc[:,cat_features] = test.iloc[:,cat_features].replace(np.nan,'nan')
model = joblib.load(os.path.join(model_path,'catboost_bst_model.pkl'))
model.best_estimator_.fit(raw_data, raw_target['status_group'])
preds = model.predict_proba(test)
best_preds = [np.argmax(line) for line in preds]
test1['status_group'] = best_preds
recode2={v:k for k,v in recode.items()}
test1['status_group']=test1['status_group'].map(recode2)
print("\nPredictions sample: \n",test1[['id','status_group']].head(5))
test1[['id','status_group']].to_csv(os.path.join(DATA_DIR,'pred1_ctb.csv'),header=True,index=False)

