# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 12:03:18 2019

@author: sursa
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
df = pd.read_csv("SP.csv")
display(df.head())

passmark=40

#size of data frame
print(df.shape)
print(df.describe())

#checking for null values
print(df.isnull().sum())

#count plot for math score 
#p = sns.countplot(x="math score", data = df, palette="muted")
#p1= plt.setp(p.get_xticklabels(), rotation=90) 

#count plot for reading score score 
#p = sns.countplot(x="reading score", data = df, palette="muted")
#_ = plt.setp(p.get_xticklabels(), rotation=90) 

#count plot for writing score score 
#p = sns.countplot(x="writing score", data = df, palette="muted")
#_ = plt.setp(p.get_xticklabels(), rotation=90) 

#number of students who passed the math exam\
df['Math_PassStatus'] = np.where(df['math score']<passmark, 'F', 'P')
print(df.Math_PassStatus.value_counts())

#number of students who passed the reading exam\
df['reading_PassStatus'] = np.where(df['reading score']<passmark, 'F', 'P')
print(df.reading_PassStatus.value_counts())

#number of students who passed the writing exam\
df['writing_PassStatus'] = np.where(df['writing score']<passmark, 'F', 'P')
print(df.writing_PassStatus.value_counts())

#count plot for parental level of education
#p = sns.countplot(x="parental level of education",data = df , hue="Math_PassStatus", palette="bright")
#_= plt.setp(p.get_xticklabels(), rotation=90)

#count plot for parental level of education_reading_score
#p = sns.countplot(x="parental level of education",data = df , hue="reading_PassStatus", palette="bright")
#_= plt.setp(p.get_xticklabels(), rotation=90)

#count plot for parental level of education_reading_score
#p = sns.countplot(x="parental level of education",data = df , hue="writing_PassStatus", palette="bright")
#_= plt.setp(p.get_xticklabels(), rotation=90)

#overall pass status 
df['OverAll_PassStatus'] = df.apply(lambda x : 'F' if x['Math_PassStatus'] == 'F' or 
                                    x['reading_PassStatus'] == 'F' or x['writing_PassStatus'] == 'F' else 'P', axis =1)

print(df.OverAll_PassStatus.value_counts())

#count plot for parental level of education_reading_score
#p = sns.countplot(x="parental level of education",data = df , hue="OverAll_PassStatus", palette="bright")
#_= plt.setp(p.get_xticklabels(), rotation=90)

#percentage of students
df['Total_Marks'] = df['math score']+df['reading score']+df['writing score']
df['Percentage'] = df['Total_Marks']/3

p = sns.countplot(x="Percentage", data = df, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=0)

#converting percentage to grades
def GetGrade(Percentage, OverAll_PassStatus):
    if ( OverAll_PassStatus == 'F'):
        return 'F'    
    if ( Percentage >= 80 ):
        return 'A'
    if ( Percentage >= 70):
        return 'B'
    if ( Percentage >= 60):
        return 'C'
    if ( Percentage >= 50):
        return 'D'
    if ( Percentage >= 40):
        return 'E'
    else: 
        return 'F'

df['Grade'] = df.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)

print(df.Grade.value_counts())
sns.countplot(x="Grade", data = df, order=['A','B','C','D','E','F'],  palette="muted")
plt.show()

#count plot for parental level of education_reading_score
p = sns.countplot(x="parental level of education",data = df , hue="Grade", palette="bright")
_= plt.setp(p.get_xticklabels(), rotation=90)