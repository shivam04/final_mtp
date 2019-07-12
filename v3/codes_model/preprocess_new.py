
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import glob, os, sys
import time
import pickle
import datetime


STATIC_FIELDS = ['ICUSTAY_ID','EMERGENCY']
FIELDS = ['ICUSTAY_ID', 'TimeStamp', 'Albumin', 'Alk. Phosphate', 'ALT', 'AST', 'Total Bili', 'BUN', 'Cholesterol',
          'Creatinine', 'Arterial BP [Diastolic]', 'FiO2 Set', 'GCS Total', 'Glucose', 'HCO3', 'Hematocrit',
          'Heart Rate', 'Potassium', 'Lactic Acid', 'Arterial BP Mean', 'MechVent', 'Magnesium', 'Sodium', 'NIDiasABP',
          'NIMAP', 'NISysABP', 'Arterial PaCO2', 'Arterial PaO2', 'Arterial pH', 'Platelets', 'Respiratory Rate',
          'SaO2', 'Arterial BP [Systolic]', 'Temperature C', 'TroponinI', 'TroponinT', 'Urine', 'WBC',
          'Previous WeightF']
df_all = pd.DataFrame(columns=FIELDS)
df_static_all = pd.DataFrame(columns=STATIC_FIELDS)


# In[4]:

file_count = 0
unq_ids = []
time_i = time.time()
for filename in glob.glob('/home/iiitm/Desktop/code original data/24 jan/bk/*.txt'):
#     if j % 100 == 0:
#         print j, str(time.time() - time_i)
    file_count += 1
    file = open(filename)
    print filename
    file_list = [] 
    # print filename
    # for line in file:
    #     file_list.append(line.strip("\n").split(","))
    # mm = -1
    # for filee in file_list:
    # 	if len(filee)==4:
    # 		print filee
    file_list_df = pd.read_csv(filename,sep=",")
    file_list = file_list_df.as_matrix()
    #file_list = file_df.as_matrix()
    #file_list_df = pd.DataFrame(file_list, columns=['Time', 'Parameter', 'Value'])

    unq_times = list(set(list(file_list_df['Time'])) - set(['Time']))
    unq_times = sorted(unq_times)
    id = file_list_df.loc[file_list_df.Parameter=='ICUSTAY_ID', ['Value']].values[0][0]
    # print id
    unq_ids.append(int(id))
    #print unq_ids
    a = np.empty((len(unq_times),len(FIELDS),))
    a[:] = np.NAN
    a[:,0] = id
    df = pd.DataFrame(a,columns=FIELDS)
    df['TimeStamp'] = unq_times
    #print len(file_list)
    for i in range(len(file_list)):
        try:
            timestamp = file_list[i][0]
            var = file_list[i][1]
            val = file_list[i][2]
            df.loc[(df.TimeStamp == timestamp), [var]] = val
           # print 'j',var,val
        except KeyError:
			#print "maaaka"
			continue
    # age = file_list_df.loc[file_list_df.Parameter=='Age', ['Value']].values[0][0]
    # Gender = file_list_df.loc[file_list_df.Parameter=='Gender', ['Value']].values[0][0]
    # Height = file_list_df.loc[file_list_df.Parameter=='Height', ['Value']].values[0][0]
    # ICUType = file_list_df.loc[file_list_df.Parameter=='ICUType', ['Value']].values[0][0]  
    EMERGENCY = file_list_df.loc[file_list_df.Parameter=='Emerygency', ['Value']].values[0][0]   
    df_static = pd.DataFrame(columns=STATIC_FIELDS)
    data = pd.DataFrame([[int(id), EMERGENCY]], columns=STATIC_FIELDS)
    df_static =  df_static.append(data)
    df_static_all = df_static_all.append(df_static)
    df_all = df_all.append(df)
# print df


# # In[48]:

unq_ids[0]



# # In[5]:

a = [df_all,  df_static_all]
with open('filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol = pickle.HIGHEST_PROTOCOL)


# # In[6]:
with open('filename.pickle', 'rb') as handle:
    [df_all,  df_static_all] = pickle.load(handle)
    print "yes"
print "here"

# # In[7]:

df_all['TimeStamp'] = df_all['TimeStamp'] + ':00'
df_all['TimeStamp'] = pd.to_timedelta(df_all['TimeStamp'])
df_all = df_all.reset_index(drop=True)


# # In[8]:

time_i = time.time()
i = 0
for id_ in unq_ids:
    if i%100 == 0:
        print (i, time_i - time.time())
    i = i+1
    id1 = np.empty((1,39))
    id1 = id1.astype(str)
    id1[:] = np.NAN
    id1[0][0] = id_
    id1[0][1] = '48:59:59'
    df_add = pd.DataFrame(id1, columns=df_all.columns)
    df_all = df_all.append(df_add)


# In[9]:

df_all['TimeStamp'] = pd.to_timedelta(df_all['TimeStamp'])
df_all = df_all.reset_index(drop=True)


# In[10]:

df_all['ICUSTAY_ID'] = df_all['ICUSTAY_ID'].astype('int')
df_all[df_all.columns[2:]] = df_all[df_all.columns[2:]].astype('float')


# In[11]:

df_all.head()


# In[12]:

feat_df = pd.DataFrame(columns=df_all.columns)
time_i = time.time()
i = 0
for id_ in unq_ids:
    if i%100 == 0:
        print (i, time_i - time.time())
    i = i+1
    feat_id = df_all.loc[df_all['ICUSTAY_ID'] == int(id_)].resample('1H', on='TimeStamp').agg(np.nanmean)
    feat_id['ICUSTAY_ID'] = int(id_) 
    feat_df = feat_df.append(feat_id)


# In[13]:

feat_df['TimeStamp'] = feat_df.index


# In[14]:

feat_df = feat_df.reset_index(drop=True)


# In[15]:

feat_df.loc[feat_df.ICUSTAY_ID==134389.0]


# In[16]:

feat_df.shape


# In[17]:

with open('feat_df.pickle', 'wb') as handle:
    pickle.dump(feat_df, handle, protocol = pickle.HIGHEST_PROTOCOL)


# In[18]:

feat_df_backfill = pd.DataFrame(columns=feat_df.columns)
time_i = time.time()
i = 0
for id_ in unq_ids:
    if i%100 == 0:
        print (i, time_i - time.time())
    i = i+1
    feat_id_backfill = feat_df.loc[feat_df['ICUSTAY_ID'] == int(id_)].fillna(method='bfill')
    #feat_id_backfill = feat_df.loc[df_all['ICUSTAY_ID'] == int(id_)].fillna(method='ffill')
    feat_df_backfill = feat_df_backfill.append(feat_id_backfill)


# In[19]:

feat_df_ffill = pd.DataFrame(columns=feat_df.columns)
time_i = time.time()
i = 0
for id_ in unq_ids:
    if i%100 == 0:
        print (i, time_i - time.time())
    i = i+1
    #feat_id_backfill = feat_df.loc[df_all['ICUSTAY_ID'] == int(id_)].fillna(method='bfill')
    feat_id_ffill = feat_df_backfill.loc[feat_df_backfill['ICUSTAY_ID'] == int(id_)].fillna(method='ffill')
    feat_df_ffill = feat_df_ffill.append(feat_id_ffill)


# In[35]:

normal_val = [-1]*39


# In[36]:

i = 2
for col in feat_df_ffill.columns[2:]:
    #print i
    feat_df_ffill[col] = feat_df_ffill[col].fillna(normal_val[i])
    i = i+1


# In[37]:

from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(feat_df_ffill[feat_df_ffill.columns[2:]])


# In[38]:

feat_df_ffill[feat_df_ffill.columns[2:]] = X


# In[72]:

feat_df_ffill1 = feat_df_ffill.sort_values(['ICUSTAY_ID','TimeStamp'], ascending=[True,True])
feat_df_ffill1 = feat_df_ffill1.reset_index(drop=True)


# In[57]:

outcomes = pd.read_csv("/home/iiitm/Desktop/code original data/24 jan/mortality1.csv")


# In[74]:

outcomes = outcomes.sort_values(['ICUSTAY_ID'])


# In[76]:

matrix3D = np.array(feat_df_ffill1.drop(['ICUSTAY_ID', 'TimeStamp'], 1))


# In[77]:
print matrix3D.shape
matrix3D = np.array(matrix3D).reshape((len(unq_ids), 49, 37))


# In[79]:

with open('preprocess.pickle', 'wb') as handle:
    pickle.dump([matrix3D, outcomes], handle, protocol = pickle.HIGHEST_PROTOCOL)


# In[81]:

outcomes.to_csv("preprocess/outcomes.csv", index=False)


# In[86]:

feat_df_ffill1.to_csv("preprocess/feat_df_ffill1.csv", index=False)


# In[87]:

# model.predict(trainX)


# # In[ ]:



