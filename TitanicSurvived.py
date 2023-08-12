import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np

dataset=pd.read_csv("train.csv")

print(dataset.columns)

#Column Selection/Field Selection
#y=dataset[["Survived"]]
#print(y.head())

gender=dataset['Sex']
sns.countplot(data=dataset,x='Sex')
sns.countplot(data=dataset,x='Sex',hue='Survived')

sns.countplot(data=dataset,x='Pclass')
sns.countplot(data=dataset,x='Pclass',hue='Survived')

sns.countplot(data=dataset,x='SibSp')
sns.countplot(data=dataset,x='SibSp',hue='Survived')

sns.countplot(data=dataset,x='Parch')
sns.countplot(data=dataset,x='Parch',hue='Survived')

sns.countplot(data=dataset,x='Embarked')
sns.countplot(data=dataset,x='Embarked',hue='Survived')

dataset.drop("PassengerId",axis=1,inplace=True) #drop passengerid
dataset.drop("Name",axis=1,inplace=True) #drop Name
dataset.drop("Ticket",axis=1,inplace=True) #drop Ticket
dataset.drop("Fare",axis=1,inplace=True) #drop Fare
dataset.drop("Parch",axis=1,inplace=True) #drop parch
dataset.drop("Cabin",axis=1,inplace=True) #drop Cabin

#x=dataset[['Pclass','Sex','Age','SibSp','Cabin','Embarked']]

#Working with Null values
print(dataset.isnull())
sns.heatmap(dataset.isnull())
sns.heatmap(dataset.isnull() ,yticklabels=False , cmap="YlGnBu")
print(dataset[dataset["Age"].isnull()]) #print null value

#here we find out the mean value using the perticular pclass
for i in range(1,4):
    age=int(dataset[dataset["Pclass"]==i]['Age'].dropna().mean())
    print(age)

#fill the Null value
def set_age(row):
    Pclass=row[0]
    age=row[1]
    if np.isnan(age):
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 29
        else:
            return 25
    else:
        return age
    

dataset[['Pclass','Age']].apply(set_age,axis=1)

# Working on Catagorical variable
#creating dummy variable
Pclass=pd.get_dummies(dataset['Pclass'],drop_first=True)
Pclass.head()

Sex=pd.get_dummies(dataset['Sex'],drop_first=True)
Sex.head()

SibSp=pd.get_dummies(dataset['SibSp'],drop_first=True)
SibSp.head()

Embarked=pd.get_dummies(dataset['Embarked'],drop_first=True)
Embarked.head()

#we have to drop catagorical variable
dataset.drop("Pclass",axis=1,inplace=True)
dataset.drop("Sex",axis=1,inplace=True)
dataset.drop("SibSp",axis=1,inplace=True)
dataset.drop("Embarked",axis=1,inplace=True)

y=dataset[['Survived']]
dataset.drop("Survived",axis=1,inplace=True)

#use that variable that we have converted into dummy variable 
dataset=pd.concat([Pclass,Sex,SibSp,Embarked] , axis=1)
print(dataset.head())

#in some version of python feature name should be string
x.columns=x.columns.astype(str) 
x.columns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split( x, y, test_size=0.20, random_state=20)

print(x_train)
print(y_train)
model=LogisticRegression()
model.fit(x_train,y_train)
print(model.coef_)
print(model.intercept_)
y_pred=model.predict(x_test)
print(y_pred)

#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))