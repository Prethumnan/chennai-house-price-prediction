#!/usr/bin/env python
# coding: utf-8

# In[65]:


#Importing necessory libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics


# In[66]:


#Reading the dataset
data=pd.read_csv("train-chennai-sale.csv")
data


# In[67]:


data.isnull().sum()


# In[68]:


#Dropping all the columns which have null values
data.dropna(inplace=True)


# In[69]:


data["AREA"].value_counts()


# In[70]:


#Treating duplicates
data["AREA"]=data["AREA"].replace({"Chrompt":"Chrompet","Chormpet":"Chrompet","Chrmpet":"Chrompet","TNagar":"T Nagar","Ana Nagar":"Anna Nagar","Karapakam":"Karapakkam","Ann Nagar":"Anna Nagar","Velchery":"Velachery","Adyr":"Adyar","KKNagar":"KK Nagar"})


# In[71]:


data["SALE_COND"].value_counts()


# In[72]:


data["SALE_COND"]=data["SALE_COND"].replace({"Adj Land":"AdjLand","Ab Normal":"AbNormal","Partiall":"Partial","PartiaLl":"Partial"})


# In[73]:


data["PARK_FACIL"].value_counts()


# In[74]:


data["PARK_FACIL"]=data["PARK_FACIL"].replace({"Noo":"No"})


# In[75]:


data["BUILDTYPE"].value_counts()


# In[76]:


data["BUILDTYPE"]=data["BUILDTYPE"].replace({"Other":"Others","Comercial":"Commercial"})


# In[77]:


data["UTILITY_AVAIL"].value_counts()


# In[78]:


data["UTILITY_AVAIL"]=data["UTILITY_AVAIL"].replace({"All Pub":"AllPub","NoSewr":"NoSeWa"})


# In[79]:


data["STREET"].value_counts()


# In[80]:


data["STREET"]=data["STREET"].replace({"Pavd":"Paved","NoAccess":"No Access"})


# In[81]:


data["MZZONE"].value_counts()


# In[82]:


age1=[]
age2=[]
for i in data["DATE_SALE"]:
    i=i.split("-")
    age1.append(int(i[-1]))
for i in data["DATE_BUILD"]:
    i=i.split("-")
    age2.append(int(i[-1]))
age=[]
for i in range(len(age1)):
    age.append(age1[i]-age2[i])
data["AGE"]=age


# In[83]:


#Checking is there any relation between distance from road to home
m=[]
for i in data["AREA"]:
    if i not in m:
        m.append(i)
for i in m:
    v=data[data["AREA"]==i]["DIST_MAINROAD"].mean()
    print(i,v)


# In[128]:


sns.distplot(data["SALES_PRICE"])


# In[84]:


plt.scatter(data["DIST_MAINROAD"],data["SALES_PRICE"])


# In[85]:


plt.scatter(data["INT_SQFT"],data["SALES_PRICE"])


# In[86]:


plt.scatter(data["QS_OVERALL"],data["SALES_PRICE"])


# In[87]:


plt.scatter(data["AGE"],data["SALES_PRICE"])


# In[88]:


data[["AREA","SALES_PRICE"]].groupby("AREA").mean().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[89]:


data[["SALE_COND","SALES_PRICE"]].groupby("SALE_COND").sum().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[90]:


data[["AREA","SALES_PRICE"]].groupby("AREA").sum().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[91]:


data.drop("DIST_MAINROAD",axis=1,inplace=True)


# In[92]:


data[["N_BEDROOM","SALES_PRICE"]].groupby("N_BEDROOM").sum().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[93]:


data[["N_BATHROOM","SALES_PRICE"]].groupby("N_BATHROOM").sum().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[94]:


data[["N_ROOM","SALES_PRICE"]].groupby("N_ROOM").sum().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[95]:


data.drop(["N_BEDROOM","N_BATHROOM"],axis=1,inplace=True)


# In[96]:


data.drop(["COMMIS","REG_FEE","PRT_ID"],axis=1,inplace=True)


# In[97]:


data[["SALE_COND","SALES_PRICE"]].groupby("SALE_COND").sum().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[98]:


data[["SALE_COND","SALES_PRICE"]].groupby("SALE_COND").sum().sort_values(by=["SALES_PRICE"])


# In[99]:


data.drop("SALE_COND",axis=1,inplace=True)


# In[100]:


data[["BUILDTYPE","SALES_PRICE"]].groupby("BUILDTYPE").sum().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[101]:


data[["BUILDTYPE","SALES_PRICE"]].groupby("BUILDTYPE").sum().sort_values(by=["SALES_PRICE"])


# In[102]:


data[["UTILITY_AVAIL","SALES_PRICE"]].groupby("UTILITY_AVAIL").sum().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[103]:


data[["UTILITY_AVAIL","SALES_PRICE"]].groupby("UTILITY_AVAIL").sum().sort_values(by=["SALES_PRICE"])


# In[104]:


statistics.variance(data[["UTILITY_AVAIL","SALES_PRICE"]].groupby("UTILITY_AVAIL").sum().sort_values(by=["SALES_PRICE"])["SALES_PRICE"])


# In[105]:


data.drop("UTILITY_AVAIL",axis=1,inplace=True)


# In[106]:


data.drop(["DATE_SALE","DATE_BUILD"],axis=1,inplace=True)


# In[107]:


p=data[["PARK_FACIL","SALES_PRICE"]]
print(np.mean(p[p["PARK_FACIL"]=="Yes"]["SALES_PRICE"]))


# In[108]:


p=data[["PARK_FACIL","SALES_PRICE"]]
print(np.mean(p[p["PARK_FACIL"]=="No"]["SALES_PRICE"]))


# In[109]:


data[["STREET","SALES_PRICE"]].groupby("STREET").sum().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[110]:


statistics.variance(data[["STREET","SALES_PRICE"]].groupby("STREET").mean()["SALES_PRICE"])


# In[111]:


data[["MZZONE","SALES_PRICE"]].groupby("MZZONE").sum().sort_values(by=["SALES_PRICE"]).plot(kind="bar")


# In[112]:


data.drop(["QS_ROOMS","QS_BATHROOM","QS_BEDROOM","QS_OVERALL","AGE"],axis=1,inplace=True)


# In[113]:


#Doing encoding
data["AREA"]=data["AREA"].replace({"Chrompet":1,"KK Nagar":2,"Anna Nagar":3,"Velachery":4,"Karapakkam":5,"T Nagar":6,"Adyar":7})


# In[114]:


data["PARK_FACIL"]=data["PARK_FACIL"].replace({"Yes":1,"No":0})


# In[115]:


data["BUILDTYPE"]=data["BUILDTYPE"].replace({"Commercial":1,"Others":2,"House":3})


# In[116]:


data["MZZONE"]=data["MZZONE"].replace({"RM":1,"RL":2,"RH":3,"I":4,"C":5,"A":6})


# In[117]:


data["STREET"]=data["STREET"].replace({"Gravel":1,"Paved":2,"No Access":3})


# In[118]:


#Checking correlation 
sns.heatmap(data.corr(),annot=True,cmap="coolwarm")


# In[119]:


data.corr().sort_values(by="SALES_PRICE")


# In[120]:


data.drop(["N_ROOM"],axis=1,inplace=True)


# In[121]:


data


# In[122]:


#Going to build model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[123]:


x=data.drop("SALES_PRICE",axis=1)
y=data["SALES_PRICE"]


# In[124]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[125]:


lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("r2_score :",r2_score(y_test, y_pred))
print("Cross_val_score :",np.mean(cross_val_score(lr,x_train,y_train)))


# In[126]:


tree=DecisionTreeRegressor()
tree.fit(x_train,y_train)
y_pred=tree.predict(x_test)
print("r2_score :",r2_score(y_test, y_pred))
print("Cross_val_score :",np.mean(cross_val_score(tree,x_train,y_train)))


# In[132]:


xgb=XGBRegressor()
xgb.fit(x_train,y_train)
y_pred=tree.predict(x_test)
print("r2_score :",r2_score(y_test, y_pred))
print("Cross_val_score :",np.mean(cross_val_score(xgb,x_train,y_train)))


# In[138]:


n=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})


# In[140]:


n

