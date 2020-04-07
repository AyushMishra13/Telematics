#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Import libraries
import data_preprocessing as pre
import global_vals as global_vals
import csv
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def get_acc_roll_angle(acc_y, acc_z):
    """
    returns roll angle
    """
    return np.degrees(np.arctan(acc_y / acc_z))

def get_acc_pitch_angle(acc_x, acc_y, acc_z):
    """"
    Return Pitch angle
    """
    denominator = np.sqrt(np.power(acc_y, 2) + np.power(acc_z, 2))
    pitch_angle = np.degrees(np.arctan(((-1) * acc_x) / denominator))
    return pitch_angle


# In[3]:


#Data Pre Processing
path = r'C:/Users/Ayush.Kumar/Documents/workspace/telematics/Driving_score_prediction/data/raw_data'
# Extract all files extension is .csv
files = glob.glob(os.path.join(path, "*.csv"))
#files = files[0:11]
data = []
for csv in files:
    header = ['time_vector(s)','rpm','speed(kmph)','engine_load','acc_x(G)','acc_y(G)','acc_z(G)','gyro_x(rad_per_s)','gyro_y(rad_per_s)','gyro_z(rad_per_s)','mag_x','mag_y','mag_z']   
    frame = pd.read_csv(csv,names=header)
    frame['date'] = os.path.basename(csv)
    data.append(frame)


# In[4]:


df = pd.concat(data, ignore_index=True)
df['date'] =df['date'].apply(lambda x : x.split(".")[0].replace("_","-"))
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d-%H-%M-%S')
df = df[df['speed(kmph)'] > 0 ]  #when vehicle is moving
df.reset_index(drop=True, inplace=True)

ac = []
gy = []
for i in range(0, len(df)):
    a = max(df['acc_x(G)'][i] , df['acc_y(G)'][i] , df['acc_z(G)'][i])
    b = max(df['gyro_x(rad_per_s)'][i], df['gyro_y(rad_per_s)'][i], df['gyro_z(rad_per_s)'][i])
    ac.append(a)
    gy.append(b)

df['acc'] = ac
df['gyro'] = gy
df['accX_ms'] = df['acc_x(G)'] * 9.807
df['accY_ms'] = df['acc_y(G)'] * 9.807
df['accZ_ms'] = df['acc_z(G)'] * 9.807
df['gyroX_dps'] = df['gyro_x(rad_per_s)'] * 57.296
df['gyroY_dps'] = df['gyro_y(rad_per_s)'] * 57.296
df['gyroZ_dps'] = df['gyro_z(rad_per_s)'] * 57.296
df['acc_ms'] = df['acc']  * 9.807
df['gyro_dps'] = df['gyro'] * 57.296
df['speed_mps'] = df['speed(kmph)'] *0.278
df['acc_roll_angle'] = get_acc_roll_angle(df['accY_ms'], df['accZ_ms'])
df['acc_pitch_angle'] = get_acc_pitch_angle(df['accX_ms'], df['accY_ms'], df['accZ_ms'])
df1 = df[['date','speed_mps', 'acc_ms', 'gyro_dps', 'acc_roll_angle', 'acc_pitch_angle']]
df1.to_csv("C:/Users/mohit.awana/Documents/workspace/telematics/Driving_score_prediction/data/processed_data/processed_data.csv", index=False) #saving in csv


# In[5]:


class Driver:
    
    def __init__(self):
        #preprocess = pre.data_preprocess()
        self.dataset = df1
        self.file_init = False
        self.file_name = 'feature_outputs/'
        
    def get_dataset(self):
        return self.dataset
    
    def generate_features_csv(self,feature_vector,row_count):
        if(self.file_init == False):
            self.file_name += "driver_features.csv"
            self.file_init = True
            header = ['index','max_speed','max_rotation','max_acc','avg_acc','avg_rotation','avg_speed','min_acc','min_speed','min_rotation','acc_roll_angle', 'acc_pitch_angle']
            with open(self.file_name, 'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow(header)
            writeFile.close()
        else:
            feature_vector = list(feature_vector.values())
            feature_vector = [row_count]+feature_vector
            with open(self.file_name, 'a') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow(feature_vector)
            writeFile.close()
    
    def write_features(self,feature_vector,row_count):
        if(self.file_init == False):
            self.file_name += "driver_features.csv"
            self.file_init = True
            header = ['index','max_speed','max_rotation','max_acc','avg_acc','avg_rotation','avg_speed','min_acc','min_speed','min_rotation','acc_roll_angle', 'acc_pitch_angle','score']
            with open(self.file_name, 'w') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow(header)
            writeFile.close()
        else:
            score = 10.0
            if (feature_vector['max_speed'] > 10.5):
                score = score - 1.3
            elif (feature_vector['max_speed'] >12.5):
                score = score - 2.9
            elif (feature_vector['max_speed'] > 15.0): 
                score = score - 5.7
            if (feature_vector['avg_speed'] >10.5):
                score = score - 1.5
            #no need for other speed conditions since max conditions covers them
            if (feature_vector['max_rotation'] >0.1):
                score = score - 0.5
            elif (feature_vector['max_rotation'] >0.8):
                score = score - 0.8
            if (feature_vector['avg_rotation'] >0.1):
                score = score - 0.6
            if (feature_vector['max_acc'] >1.5 and feature_vector['max_acc'] <2.0):
                score = score - 1.8
            elif (feature_vector['max_acc'] >1.0 and feature_vector['max_acc'] <2.0):
                score = score - 1.0
            if (feature_vector['avg_acc'] >1.0):
                score = score - 0.4
            feature_vector = list(feature_vector.values())
            feature_vector = [row_count]+feature_vector+[score]
            with open(self.file_name, 'a') as writeFile:
                writer = csv.writer(writeFile)
                writer.writerow(feature_vector)
            writeFile.close()
            

    
    def generate_feature_vectors(self):
        self.data_rows = self.dataset.iloc[:,:-1].values.tolist()
        self.feature_vector = {}
        speed = self.dataset['speed_mps'].values
        acc = self.dataset['acc_ms'].values
        gyro = self.dataset['gyro_dps'].values
        acc_roll_angle = self.dataset['acc_roll_angle'].values
        acc_pitch_angle = self.dataset['acc_pitch_angle'].values
        print(len(self.data_rows))
        for row_count in range(len(self.data_rows)):
            feature_element = {}
            max_speed = 0
            min_speed = 999
            avg_speed = 0
            max_acc = 0
            min_acc =999
            avg_acc = 0
            max_rotation = 0
            min_rotation = 999
            avg_rotation = 0
            count=0
            if (row_count+global_vals.sample_size>len(self.data_rows)):
                global_vals.sample_size = global_vals.sample_size - row_count - 1
            while count in range(global_vals.sample_size):
                max_speed = abs(speed[row_count+count]) if abs(speed[row_count+count])>max_speed else max_speed
                max_rotation = abs(gyro[row_count+count]) if abs(gyro[row_count+count])>max_rotation else max_rotation
                max_acc = abs(acc[row_count+count]) if abs(acc[row_count+count])>max_acc else max_acc
                acc_roll = acc_roll_angle[row_count+count]
                acc_pitch = acc_pitch_angle[row_count+count]

                avg_speed+=abs(speed[row_count+count])
                avg_acc+=abs(acc[row_count+count])
                avg_rotation+=abs(gyro[row_count+count])
                
                min_speed = abs(speed[row_count+count]) if abs(speed[row_count+count])<min_speed else min_speed
                min_rotation = abs(gyro[row_count+count]) if abs(gyro[row_count+count])<min_rotation else min_rotation
                min_acc = abs(acc[row_count+count]) if abs(acc[row_count+count])<min_acc else min_acc
                count+=1
            
            if(count>0):
                avg_acc/=global_vals.sample_size
                avg_speed/=global_vals.sample_size
                avg_rotation/=global_vals.sample_size
                
                feature_element['max_speed'] = max_speed
                feature_element['max_rotation'] = max_rotation
                feature_element['max_acc'] = max_acc
                feature_element['avg_acc'] = avg_acc
                feature_element['avg_rotation'] = avg_rotation
                feature_element['avg_speed'] = avg_speed
                feature_element['min_acc'] = min_acc
                feature_element['min_speed'] = min_speed
                feature_element['min_rotation'] = min_rotation
                feature_element['acc_roll_angle'] = acc_roll
                feature_element['acc_pitch_angle'] = acc_pitch
                
                
                if global_vals.train_mode == True:
                    self.write_features(feature_element,row_count)
                else:
                    self.generate_features_csv(feature_element,row_count)
                
                
                self.feature_vector[row_count] = feature_element

        return self.feature_vector


# In[7]:


Driver_obj = Driver()
feature_dict = Driver_obj.generate_feature_vectors()


# In[17]:


df2 = pd.read_csv('feature_outputs/driver_features.csv')
df2 = df2.drop(['index'], axis=1)


# In[18]:


#Modeling
X = df2.iloc[:, :11].values
y = df2.iloc[:, 11].values

for count in range(0,len(y)):
    if y[count] <7.5:
        y[count]=0
    else:
        y[count]=1


# In[20]:


df2['score'].value_counts()


# In[21]:


#Modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.coef_)
print(lr.intercept_)
y_pred = lr.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[22]:


#Accuracy of the model
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(X_test, y_test)))


# In[23]:


print(classification_report(y_test, y_pred))


# In[24]:


logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test,lr.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




