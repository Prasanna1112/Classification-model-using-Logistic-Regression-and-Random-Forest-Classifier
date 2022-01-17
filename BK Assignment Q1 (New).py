#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np


df_diab = pd.read_csv('https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/diabetes.csv')

# print(df_tit.info())
print('Shape of dataframe:', df_diab.shape)
df_diab.head()


# In[26]:


# df_diab.describe().T
df_diab.info()


# In[33]:


import seaborn as sns
sns.countplot(x=df_diab['Outcome'])


# In[34]:


df_diab.isnull().sum()


# In[36]:


df_diab[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = df_diab[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)
df_diab.head() #Replaced all 0 values by NaN so it is easy to clean the data

df_diab.fillna(df_diab.mean(), inplace = True) #Filled Mising values with Mean (Imputation)
df_diab.head()


# In[52]:


from scipy import stats
from sklearn.preprocessing import MinMaxScaler

data = df_diab
col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# plot raw data
fig = plt.figure(figsize = (20, 12))
plt.subplots_adjust(left = 0.12, right = 0.95, bottom = 0.05, top = 0.95,
                    wspace = 0.35, hspace = 0.25)
plt.subplot(2, 2, 1)
plt.title('Raw correct data')
data.boxplot(vert = False, labels = col_names, patch_artist = True)

# remove outliers and plot results
plt.subplot(2, 2, 2)
plt.title('Data without outliers')
data['Insulin'] = data['Insulin'] * .001
data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
data.boxplot(vert = False, labels = col_names, patch_artist = True)

# scale data and plot results
plt.subplot(2 , 2, 3)
plt.title('Scaled Data')
scaler = MinMaxScaler(feature_range=(0, 1))
data   = pd.DataFrame(scaler.fit_transform(data.values),  columns = col_names)
data.boxplot(vert = False, labels = col_names, patch_artist = True)

# df_diab[:0]


# In[38]:


import matplotlib.pyplot as plt # Using seaborn plot to check correlation between features
sns.heatmap(df_diab.corr(),annot=True)
fig = plt.gcf()
fig.set_size_inches(8,8)


# In[40]:


# Feature selection

from sklearn.ensemble import RandomForestClassifier
RFclf = RandomForestClassifier()
x =  df_diab[df_diab.columns[:8]]
y = df_diab.Outcome
RFclf.fit(x,y)
feature_imp = pd.DataFrame(RFclf.feature_importances_,index=x.columns)
feature_imp.sort_values(by = 0 , ascending = False)


# In[58]:


from sklearn.feature_selection import RFE
from sklearn import metrics

for i in range(2,7):
    rfe = RFE(estimator=RandomForestClassifier(),n_features_to_select=i, verbose=0)
    rfe.fit(x,y)
#     print(f"Accuracy with Feature {i}: ",metrics.accuracy_score(y_test, rfe.predict(x_test_std)))

print("Important Features are: ", list(df_diab.columns[:8][rfe.support_]))


# In[63]:


from sklearn.model_selection import train_test_split

features = df_diab[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
labels = df_diab.Outcome
features.head()

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,
                                                                         stratify=df_diab.Outcome,
                                                                         shuffle = True, 
                                                                         test_size=0.2, 
                                                                         random_state=1)

# Show the Training and Testing Data
print('Shape of training feature:', features_train.shape)
print('Shape of testing feature:', features_test.shape)
print('Shape of training label:', labels_train.shape)
print('Shape of testing label:', labels_test.shape)


# In[68]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression
LRclf = LogisticRegression()
LRclf.fit(features_train,labels_train)
labels_pred = LRclf.predict(features_test)
LRclf.score(features_test,labels_test)

# Calculate accuracy, precision, recall, f1-score, and kappa score
acc = metrics.accuracy_score(labels_test, labels_pred)
prec = metrics.precision_score(labels_test, labels_pred)
rec = metrics.recall_score(labels_test, labels_pred)
f1 = metrics.f1_score(labels_test, labels_pred)

# Display confussion matrix
cm = metrics.confusion_matrix(labels_test, labels_pred)

print('Accuracy:', acc)
print('Precision:', prec)
print('Recall:', rec)
print('F1 Score:', f1)
print('Confusion Matrix:\n', cm)


# In[ ]:




