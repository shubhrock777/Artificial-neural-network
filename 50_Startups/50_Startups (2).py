import pandas as pd


#loading the dataset
startup = pd.read_csv("D:/BLR10AM/Assi/27.ANN/Datasets_ANN Assignment/50_Startups (2).csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary
description  = ["Money spend on research and development",
                "Administration",
                "Money spend on Marketing",
                "Name of state",
                "Company profit"]

d_types =["Ratio","Ratio","Ratio","Nominal","Ratio"]

data_details =pd.DataFrame({"column name":startup.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": startup.dtypes})

            #3.	Data Pre-startupcessing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of startup 
startup.info()
startup.describe()          


#rename the columns
startup.rename(columns = {'R&D Spend':'rd_spend', 'Marketing Spend' : 'm_spend'} , inplace = True)  

#data types        
startup.dtypes


#checking for na value
startup.isna().sum()
startup.isnull().sum()

#checking unique value for each columns
startup.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    


EDA ={"column ": startup.columns,
      "mean": startup.mean(),
      "median":startup.median(),
      "mode":startup.mode(),
      "standard deviation": startup.std(),
      "variance":startup.var(),
      "skewness":startup.skew(),
      "kurtosis":startup.kurt()}

EDA


# covariance for data set 
covariance = startup.cov()
covariance

# Correlation matrix 
co = startup.corr()
co

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.


####### graphistartup repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(startup.iloc[:, :])


#boxplot for every columns
startup.columns
startup.nunique()

startup.boxplot(column=['rd_spend', 'Administration', 'm_spend', 'Profit'])   #no outlier

# here we can see lVO For profit
# Detection of outliers (find limits for RM based on IQR)
IQR = startup['Profit'].quantile(0.75) - startup['Profit'].quantile(0.25)
lower_limit = startup['Profit'].quantile(0.25) - (IQR * 1.5)

####################### 2.Replace ############################
# Now let's replace the outliers by the maximum and minimum limit
#Graphical Representation
import numpy as np
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#startup['Profit']= pd.DataFrame( np.where(startup['Profit'] < lower_limit, lower_limit, startup['Profit']))

import seaborn as sns 
sns.boxplot(startup.Profit);plt.title('Boxplot');plt.show()



# rd_spend
plt.bar(height = startup.rd_spend, x = np.arange(1, 51, 1))
plt.hist(startup.rd_spend) #histogram
plt.boxplot(startup.rd_spend) #boxplot


# Administration
plt.bar(height = startup.Administration, x = np.arange(1, 51, 1))
plt.hist(startup.Administration) #histogram
plt.boxplot(startup.Administration) #boxplot

# m_spend
plt.bar(height = startup.m_spend, x = np.arange(1, 51, 1))
plt.hist(startup.m_spend) #histogram
plt.boxplot(startup.m_spend) #boxplot


#profit
plt.bar(height = startup.Profit, x = np.arange(1, 51, 1))
plt.hist(startup.Profit) #histogram
plt.boxplot(startup.Profit) #boxplot


# Jointplot

sns.jointplot(x=startup['Profit'], y=startup['rd_spend'])



# Q-Q Plot
from scipy import stats
import pylab

stats.probplot(startup.Profit, dist = "norm", plot = pylab)
plt.show() 
# startupfit is normally distributed

stats.probplot(startup.Administration, dist = "norm", plot = pylab)
plt.show() 
# administration is normally distributed


stats.probplot(startup.rd_spend, dist = "norm", plot = pylab)
plt.show() 

stats.probplot(startup.m_spend, dist = "norm", plot = pylab)
plt.show() 

#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(startup.iloc[:,[0,1,2]])
df_norm.describe()


"""
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
sta=startup.iloc[:,[3]]
enc_df = pd.DataFrame(enc.fit_transform(sta).toarray())"""

# Create dummy variables on categorcal columns

enc_df = pd.get_dummies(startup.iloc[:,[3]])
enc_df.columns
enc_df.rename(columns={"State_New York":'State_New_York'},inplace= True)

model_df = pd.concat([enc_df, df_norm, startup.iloc[:,4]], axis =1)

# Rearrange the order of the variables
model_df = model_df.iloc[:, [6, 0,1, 2, 3,4,5]]


##################################
###upport Vector Machines MODEL###
"""5.	Model Building:
5.1	Perform Artificial Neural Network on the given datasets.
5.2	Use TensorFlow keras to build your model in Python and use Neural net package in R
5.3	Briefly explain the output in the documentation for each step in your own words.
5.4	Use different activation functions to get the best model.


"""

import numpy as np


np.random.seed(10)




X= model_df.iloc[:,1:]
Y= model_df.iloc[:,0]




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 457) # 20% test data
 

from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense
import sklearn.metrics as skl_mtc
from tensorflow import keras 
import matplotlib.pyplot as plt

model = keras.models.Sequential()
model.add(keras.layers.Dense(5000, activation='relu', input_dim=6))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, kernel_initializer='uniform'))
model.compile(loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Nadam(
        learning_rate=0.009,
        beta_1=0.8,
        beta_2=0.999),metrics=["mse"])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=1,
    patience=20,
    mode='auto',
    restore_best_weights=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    verbose=1,
    mode='auto',
    min_delta=0.0005,
    cooldown=0,
    min_lr=1e-6)



# fitting model on train data
model.fit(x=x_train,y=y_train,batch_size=2,epochs=100)


# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test,y_test,verbose = 1)

# accuracy on test data set

# accuracy score on train data 
eval_score_train = model.evaluate(x_train,y_train,verbose=1)





predict_y = model.predict(x_test)


#R2-score
result = skl_mtc.r2_score(y_test, predict_y)
print(f'R2-score in test set: {np.round(result, 4)}')

# test residual values 

# accuracy on train data set 

pred_df = pd.DataFrame(predict_y, columns =['predict_y'])
pred_y= pred_df.iloc[:,0]

test_resid = pred_y - y_test
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse



#graph for eppchs
history = model.fit(x_train, y_train, epochs=10, batch_size=2,  verbose=1, validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()