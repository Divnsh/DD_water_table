import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import datetime
from scipy.spatial import distance
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import paired_distances
from sklearn.preprocessing import MinMaxScaler
#from fancyimpute import KNN
from sklearn.preprocessing import LabelEncoder

np.random.seed(2020)
time0=datetime.datetime.now()

# Reading encoded data

DATA_DIR = '/home/divyansh/DataScience/kaggle_practice/Drivendata_competitions/Pump_it_up_water_table/Data'

data_folder = os.path.join(DATA_DIR,'*.csv')
file_list = sorted(glob(data_folder))

# Reading data
labels = pd.read_csv(file_list[0])

data=pd.read_csv(os.path.join(DATA_DIR,'raw_data_encoded.csv'))
data = data.merge(labels)
data = data.replace(-99,np.nan)

#data=data.sample(frac=1,random_state=2020) #shuffling our data before split
#train_x,val_x,train_y,val_y=train_test_split(data.iloc[:,:-1],data.iloc[:,-1],test_size=0.2)

k=13 # No of neighbours
l=0.136 # Nan penalisation factor

def dist_with_miss(a,b,l=l,missing_values=np.nan): # Custom distance metric with flexible weights on nan's
    if(len(a) != len(b)):
        return np.inf
    ls = l * np.ones(len(a))
    msk = ~ (np.isnan(a) | np.isnan(b))
    res = np.sum((np.abs(a-b)[msk]))+np.sum((ls[~msk]))
    return res

#train_x=MinMaxScaler().fit_transform(train_x)  # Scaling for knn imputation
#val_x=MinMaxScaler().fit_transform(val_x)  # Scaling for knn imputation

imputer=KNNImputer(missing_values=np.nan,n_neighbors=k,weights='uniform',metric=dist_with_miss)  # Memory hog

#imputer=KNN(k=k) # Not using custom distance

#if __name__=='__main__':
#print("KNN imputation started..")
#imputer.fit(train_x)
#train_x=imputer.transform(train_x)
# val_x=imputer.transform(val_x)
# print("Imputation complete!")
# time1=datetime.datetime.now()
# print(f'Time taken for imputation: {str(time1-time0)}')
le=LabelEncoder()
# le.fit(train_y)
# train_y=le.transform(train_y)
# val_y=le.transform(val_y)


