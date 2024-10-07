#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width',1000)


# ## Değişkenler
# customerID: Benzersiz müşteri kimliği.
# gender: Cinsiyet (Male/Female).
# SeniorCitizen: Yaşlı olup olmadığını gösterir (0: Genç, 1: Yaşlı).
# Partner: Partneri olup olmadığını belirtir (Yes/No).
# Dependents: Bağımlı kişinin olup olmadığını gösterir (Yes/No).
# tenure: Şirkette kalma süresi (ay).
# PhoneService: Telefon hizmeti alıp almadığını belirtir (Yes/No).
# MultipleLines: Birden fazla telefon hattı olup olmadığını gösterir.
# InternetService: İnternet hizmet türü (DSL/Fiber/No).
# OnlineSecurity: Çevrimiçi güvenlik hizmeti var mı (Yes/No).
# OnlineBackup: Çevrimiçi yedekleme hizmeti var mı (Yes/No).
# DeviceProtection: Cihaz koruma hizmeti var mı (Yes/No).
# TechSupport: Teknik destek hizmeti alıyor mu (Yes/No).
# StreamingTV: TV yayın akışı hizmeti alıyor mu (Yes/No).
# StreamingMovies: Film yayını hizmeti alıyor mu (Yes/No).
# Contract: Sözleşme türü (Month-to-month/One year/Two year).
# PaperlessBilling: Kağıtsız faturalandırma kullanımı (Yes/No).
# PaymentMethod: Ödeme yöntemi (Credit card/Bank transfer/Electronic check/Mailed check).
# MonthlyCharges: Aylık ücret (sayısal).
# TotalCharges: Toplam ücret (sayısal).
# Churn: Müşterinin hizmeti bırakıp bırakmadığını gösterir (Yes/No).

# In[3]:


df_= pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[43]:


df= df_.copy()
df.head()


# In[5]:


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Unique #####################")
    print(dataframe.nunique())

check_df(df)


# In[8]:


df['Recency'] = df['tenure']


# In[11]:


df['Frequency'] = df['Contract']


# In[24]:


df['Monetary'] = df['TotalCharges']


# In[26]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[31]:


rfm = df.groupby('customerID').agg({
    'tenure': 'max',  
    'Contract': lambda x: x.nunique(),
    'TotalCharges': 'sum'})
rfm.head()


# In[32]:


rfm.columns = ['recency', 'frequency', 'monetary']

rfm = rfm[rfm["monetary"] > 0]

rfm.describe().T


# In[30]:


rfm.shape


# In[34]:


rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))


# In[35]:


rfm[rfm["RFM_SCORE"] == "55"]


# In[36]:


rfm[rfm["RFM_SCORE"] == "11"]


# ## RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)

# In[37]:


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


# In[38]:


rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


# In[39]:


rfm[rfm["segment"] == "cant_loose"].head()


# In[40]:


rfm[rfm["segment"] == "cant_loose"].index


# In[41]:


rfm[rfm["segment"] == "at_Risk"].index


# In[42]:


rfm[rfm["segment"] == "at_Risk"].head()

