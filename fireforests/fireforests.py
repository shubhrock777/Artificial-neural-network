import pandas as pd
import numpy as np


#loading the dataset
forest = pd.read_csv("D:/BLR10AM/Assi/27.ANN/Datasets_ANN Assignment/fireforests.csv")


#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary




data_details =pd.DataFrame({"column name":forest.columns,
                            "data type(in Python)": forest.dtypes})

            #3.	Data Pre-forestcessing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of forest 
forest.info()
forest.describe()          

forest.nunique()



#data types        
forest.dtypes


#checking for na value
forest.isna().sum()
forest.isnull().sum()


#checking unique value for each columns
forest.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """

    


EDA ={"column ": forest.columns,
      "mean": forest.mean(),
      "median":forest.median(),
      "mode":forest.mode(),
      "standard deviation": forest.std(),
      "variance":forest.var(),
      "skewness":forest.skew(),
      "kurtosis":forest.kurt()}

EDA





# covariance for data set 
covariance = forest.cov()
covariance

# Correlation matrix 
Correlation = forest.corr()
Correlation

# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.

#variance for each column
forest.var()                   #rain column has low variance 



#droping rain colunm and month and day columns  due to thoes are already present in dummy
forest.drop(["month","day"], axis = 1, inplace = True)



####### graphidf repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(forest.iloc[:,0:9])


#boxplot for every columns
forest.columns
forest.nunique()

#boxplot for every column

# Boxplot of independent variable distribution for each category of size_category


forest.boxplot(column=['FFMC', 'DMC', 'DC', 'ISI','temp', 'RH', 'wind', 'area','rain'])  



#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(forest.iloc[:,0:8])
df.describe()


#final dataframe
model_df = pd.concat([forest.iloc[:,[8]],df,forest.iloc[:,9:28] ], axis =1)




##################################
###upport Vector Machines MODEL###
"""5.	Model Building:
5.1	Perform Artificial Neural Network on the given datasets.
5.2	Use TensorFlow keras to build your model in Python and use Neural net package in R
5.3	Briefly explain the output in the documentation for each step in your own words.
5.4	Use different activation functions to get the best model.


"""



from tensorflow.keras import Sequential
from tensorflow.keras.layers import  Dense

np.random.seed(10)




X= model_df.iloc[:,1:]
Y= model_df.iloc[:,0]




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 457) # 20% test data
 


import sklearn.metrics as skl_mtc
from tensorflow import keras 
import matplotlib.pyplot as plt

model = keras.models.Sequential()
model.add(keras.layers.Dense(5000, activation='relu', input_dim=27))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, kernel_initializer='uniform'))
model.compile(loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Nadam(
        learning_rate=0.0005,
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
model.fit(x=x_train,y=y_train,batch_size=15,epochs=100)


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
history = model.fit(x_train, y_train, epochs=10, batch_size=50,  verbose=1, validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()