import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

#load and merge dataframe and impute missing data

dftemp = pd.read_csv("dengue_features_train_updated.csv")	#load train features in a dataframe 
label = pd.read_csv("dengue_labels_train_updated.csv")	#load labels in a dataframe
df = pd.merge(dftemp,label)	#merge both the dataframes
df = df.fillna(df.mean(),inplace=True) #fill missing values with the means

#print df
#df = df.dropna()
#print df

#Data manipulation
df['time'] = (df['year'].map(str)+df['week'].map(str).apply(lambda x: x.zfill(2))).map(int) - 199018 #create a time field for reference
df['m_ndvi'] = (df['ndvi_ne']+df['ndvi_se']+df['ndvi_sw']+df['ndvi_nw'])/4	#mean of all vegetation index
df = df.drop(['date','week','year','time','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw'],axis=1)	#drop the unrequired features like sepereated vagetation indexes and date,week and also the time field as of now because later the time field becomes irrevelant
df1 = df[df['city']=="sj"]	#split the dataframes by cities
df2 = df[df['city']=="iq"]

#Data Visualization
#g = sns.heatmap(df1.corr(),annot=True,fmt=".1f")	#Visualize the heatmap of the Correlation matrix for SJ city
#g.set_xticklabels(g.get_xticklabels(),rotation = 90)	#Change the label rotation
#g.set_yticklabels(g.get_yticklabels(),rotation = 0)
#sns.plt.show()
#g = sns.heatmap(df2.corr(),annot=True,fmt=".1f")	#Visualize the heatmap of the Correlation matrix for IQ city
#g.set_xticklabels(g.get_xticklabels(),rotation = 90)	#Change the label rotation
#g.set_yticklabels(g.get_yticklabels(),rotation = 0)
#sns.plt.show()
#barplt=abs(df1.corr()).total_cases.drop('total_cases').sort_values().plot.barh() #Bar plot of all feature Correlations against 'Total Cases' for SJ city
#corrtcasesj = abs(df1.corr()).total_cases.drop('total_cases').sort_values()
#print corrtcasesj #also print the correlations
#sns.plt.show()
#barplt=abs(df2.corr()).total_cases.drop('total_cases').sort_values().plot.barh()	#Bar plot of all feature Correlations against 'Total Cases' for IQ city
#corrtcaseiq = abs(df2.corr()).total_cases.drop('total_cases').sort_values()
#print corrtcaseiq
#sns.plt.show()

#Feature Selection
listfeatures=['sp_humid','dew_tmp','st_avg_tmp','min_tmp','mean_tmp','avg_tmp','rel_humid','ppt_kg_per_m2','max_tmp','st_min_tmp','st_max_tmp']	#From the bar plot selecting the Features that are most prominent for both the cities

sj_final = df1[listfeatures]	#create a dataframe with only selected features
iq_final = df2[listfeatures]	#for both the cities

sj_label = df1['total_cases']	#also create the labels for them
iq_label = df2['total_cases']	#for both the cities

#Data Modelling
sj_features_train, sj_features_test, sj_labels_train, sj_labels_test = train_test_split(sj_final, sj_label, test_size=0.2, shuffle = False)	#split the features dataframe and labels into train and cross-validation test set of 8:2 ratio

poly = PolynomialFeatures(degree=1)	
sj_features_train_ = poly.fit_transform(sj_features_train)	#transform dataset to training vector
sj_features_test_ = poly.fit_transform(sj_features_test)
regressor = LinearRegression()

# Training our model
regressor.fit(sj_features_train_, sj_labels_train)
sj_pred = regressor.predict(sj_features_test_)
r_2_score = r2_score(sj_labels_test, sj_pred)
print r_2_score


#print sj_final
#print iq_final


#sns.heatmap(df1.corr(),annot=True)
#x = df1['time']
#y = df1['ndvi_ne']
#print x
#print y
#plt.bar(x,y)
#plt.show()
#print df.head()
#print df1.head()
#print "\n"
#print df2.head()