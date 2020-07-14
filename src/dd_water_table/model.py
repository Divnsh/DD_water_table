import xgboost as xgb
from sklearn.datasets import dump_svmlight_file
import os
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
from sklearn.pipeline import Pipeline
import datetime
from sklearn.preprocessing import MinMaxScaler
from dd_water_table.imputation import imputer,le,data
np.random.seed(2020)

model_path = '/home/divyansh/DataScience/kaggle_practice/Drivendata_competitions/Pump_it_up_water_table/model'
DATA_DIR = '/home/divyansh/DataScience/kaggle_practice/Drivendata_competitions/Pump_it_up_water_table/Data'
if not os.path.exists(model_path):
    os.makedirs(model_path)

#dump_svmlight_file(train_x, train_y, os.path.join(model_path,'dtrain.svm'), zero_based=True)
#dump_svmlight_file(val_x, val_y, os.path.join(model_path,'dtest.svm'), zero_based=True)

#dtrain_svm = xgb.DMatrix(os.path.join(model_path,'dtrain.svm'))
#dtest_svm = xgb.DMatrix(os.path.join(model_path,'dtest.svm'))

print("Data shape: ",data.shape)
train_x=np.array(data.iloc[:,:-1])
train_y=np.array(data.iloc[:,-1])
train_y = le.fit_transform(train_y)

params = {
    'm__max_depth': [3,5,7,10],  # the maximum depth of each tree
    'm__learning_rate':[0.1],
    'm__silent': [1],  # logging mode - quiet
    'm__objective': ['multi:softprob'],  # error evaluation for multiclass training
    'm__num_class': [3],
    'm__n_estimators': [100,1000],
    'm__min_child_weight' : [1,3,5,7,9],
    'm__tree_method':['gpu_hist'], # Running on GPU
    'm__seed': [2020]}  # the number of classes that exist in this datset

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2020)
model=xgb.XGBClassifier()
scaler=MinMaxScaler()

# Scaling, imputation and training pipeline
pipeline = Pipeline(steps=[('s',scaler), ('i', imputer), ('m', model)])

print("Performing 10-fold CV on train data..")
time0=datetime.datetime.now()
grid_search = GridSearchCV(pipeline, param_grid=params, scoring='accuracy', n_jobs=-1, cv=cv, verbose=3)
grid_search.fit(train_x, train_y)
time1=datetime.datetime.now()
print(f'Time taken for model fitting: {str(time1-time0)}')

print('Raw accuracy score:', grid_search.best_score_)
for param_name in sorted(grid_search.best_params_.keys()):
    print("%s: %r" % (param_name, grid_search.best_params_[param_name]))

preds = grid_search.predict_proba(train_x)
best_preds = [np.argmax(line) for line in preds]

print("\nValidation results: \n",classification_report(train_y,best_preds))

from sklearn.externals import joblib

joblib.dump(grid_search, os.path.join(model_path,'xgboost_bst_model.pkl'), compress=True)
# bst = joblib.load('bst_model.pkl') # load it later


# Reading test data and making predictions
dlist=read_Data(DATA_DIR)
test_data = dlist[3]
del dlist
main(raw_data,raw_target,'test')
test_data1=pd.read_csv(os.path.join(DATA_DIR,'test_data_encoded.csv'))
test_data =test_data1.drop('id',axis=1)

model = joblib.load('xgboost_bst_model.pkl')
preds = grid_search.predict_proba(test_data)
best_preds = [np.argmax(line) for line in preds]
test_data1['status_group'] = best_preds
print("\nPredictions sample: \n",test_data1[['id','status_group']].head(5))
test_data1[['id','status_group']].to_csv(os.path.join(DATA_DIR,'pred2_xg.csv'),header=True,index=False)