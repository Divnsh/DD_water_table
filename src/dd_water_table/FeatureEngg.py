import pandas as pd
import numpy as np
import os
from glob import glob
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)
import matplotlib.pyplot as plt
#import category_encoders as ce
import datetime
#from scipy.spatial import distance
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = '/home/divyansh/DataScience/kaggle_practice/Drivendata_competitions/Pump_it_up_water_table/Data'
pd.set_option('display.max_columns', 500)

def read_Data(DATA_DIR):
    data_folder = os.path.join(DATA_DIR,'*.csv')
    file_list = sorted(glob(data_folder))

    dlist = []

    # Reading data
    for f in file_list:
        file = pd.read_csv(f)
        dlist.append(file)
        del file

    print("All data read.")
    print("\nTarget values per id: \n",dlist[0].head())
    print("\nRaw train data: \n",dlist[1].head())
    print("\nShape of raw data: ",dlist[1].shape)
    return dlist


np.random.seed(2020)

def enc_with_na(data, encoder, n_feat):  # Helper function for encoding with nan's given the encoder and data
    enc = np.empty((data.shape[0], n_feat))
    enc[:] = -99
    nonulls = data.dropna()
    if nonulls.shape == data.shape:
        # print("No nan's")
        f = encoder.fit_transform(np.array(nonulls.astype('str')))
        return f.toarray()
    f = encoder.fit_transform(nonulls)
    enc[np.array(data.notnull()).ravel()] = f.toarray()
    return enc

def main(rawdata,rawtarget=pd.DataFrame(),train_test_flag='train'):
    raw_target = rawtarget
    raw_data = rawdata
    raw_data['date_recorded'] = pd.to_datetime(raw_data['date_recorded']).apply(
        lambda x: (datetime.datetime.today() - x).days)
    numeric_cols = [c for c in raw_data.columns if
                    raw_data[c].dtype in ['int64', 'float64'] and c not in ['region_code', 'district_code']]
    cat_cols = [c for c in raw_data.columns if c not in numeric_cols]

    # sns.pairplot(raw_data.merge(raw_target)[numeric_cols+['status_group']].iloc[:,1:], hue="status_group",diag_kind='hist',plot_kws= {'alpha': 0.5})
    # plt.show()

    # Categorical columns cardinality
    print("\n# Unique values in each categorical column:\n", raw_data[cat_cols].nunique(axis=0))

    # No. of unknown categories
    (raw_data[cat_cols] == 'unknown').sum()
    raw_data[cat_cols] = raw_data[cat_cols].replace('unknown', np.nan)

    # Deleting unneeded columns
    to_be_del = ['waterpoint_type_group', 'source_type', 'quantity_group', 'quality_group', 'payment_type',
                 'management_group', 'extraction_type_class', 'extraction_type_group', 'scheme_name', 'recorded_by',
                 'region', 'scheme_management']

    raw_data = raw_data.drop(to_be_del, axis=1)

    # % of missing values per column
    print("\nMissing value % \n", (raw_data.isna().sum() * 100 / len(raw_data)).sort_values(ascending=False))
    # (raw_data.isna().sum()*100/len(raw_data)).sort_values(ascending=False).plot(kind='bar')
    # plt.xticks(rotation=45)
    # plt.show()

    # Columns without missing values are hash encoded in bulk
    # Rest of the columns are individually hash encoded
    # This is done to preserve nan's across encoding in order to perform imputation later.

    print("Encoding categorical features..")

    ohc = ['public_meeting', 'permit', 'source_class']
    hashc_ind = ['payment', 'installer', 'funder', 'public_meeting', 'permit', 'water_quality', 'quantity',
                 'management', 'subvillage', 'source_class', 'source']
    hashc0 = ['district_code', 'region_code', 'ward', 'wpt_name', 'lga']  # 1024 bit encoding
    hashc1 = ['extraction_type', 'waterpoint_type']  # 32 bit encoding
    hashc2 = ['basin']  # 8 bit encoding

    # One hot encoding on binary categorical data
    oh = []
    for oc in ohc:
        ohe = OneHotEncoder(drop='first')
        enc = enc_with_na(raw_data[[oc]], ohe, 1)
        oh.append(enc)
    oh = np.hstack(oh).astype(np.int8)

    # Hash encoding on the rest

    # Individual hashing
    #n_feats_ind = [4, 128, 128, 6, 2, 8, 1024, 8]
    n_feats_ind = [4, 16, 16, 6, 2, 8, 64, 8]
    n_feats_ind = [4, 8, 8, 4, 2, 4, 32, 4]
    hashed_ind = []
    for hc, n in zip(hashc_ind, n_feats_ind):
        h = FeatureHasher(n_features=n, input_type='string', alternate_sign=False)
        enc = enc_with_na(raw_data[[hc]], h, n)
        hashed_ind.append(enc)
    hashed_ind = np.hstack(hashed_ind).astype(np.int8)

    # Collective hashing
    hash_cols_list = [hashc0, hashc1, hashc2]
    #n_feats = [1024, 32, 8]
    n_feats = [64, 16, 8]
    n_feats = [32, 8, 4]
    hashed = []
    for hc, n in zip(hash_cols_list, n_feats):
        h = FeatureHasher(n_features=n, input_type='string', alternate_sign=False)
        enc = enc_with_na(raw_data[hc], h, n)
        hashed.append(enc)
    hashed = np.hstack(hashed).astype(np.int8)

    print("Encoding complete..")

    print("Preparing to write data to disk..")
    raw_data_encoded = pd.concat(
        [raw_data.drop(ohc + hashc_ind + hashc0 + hashc1 + hashc2, axis=1), pd.DataFrame(np.hstack([oh, hashed_ind, hashed]))],
        axis=1)
    raw_data_encoded.to_csv(os.path.join(DATA_DIR, train_test_flag+'_data_encoded.csv'), header=True, index=False)
    print("Written encoded data to disk..")

if __name__=='main':
    dlist = read_Data(DATA_DIR)
    raw_target = dlist[0]
    raw_data = dlist[1]
    del dlist
    train_test_flag='train'
    main(raw_data,raw_target,train_test_flag)