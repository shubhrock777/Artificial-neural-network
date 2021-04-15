import pandas as pd
import numpy as np


#loading the dataset
concrete = pd.read_csv("D:/BLR10AM/Assi/27.ANN/Datasets_ANN Assignment/concrete.csv")


#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

#######feature of the dataset to create a data dictionary




data_details =pd.DataFrame({"column name":concrete.columns,
                            "data type(in Python)": concrete.dtypes})

            #3.	Data Pre-concretecessing
          #3.1 Data Cleaning, Feature Engineering, etc
          
          
#details of concrete 
concrete.info()
concrete.describe()          

concrete.nunique()



#data types        
concrete.dtypes


#checking for na value
concrete.isna().sum()
concrete.isnull().sum()


#checking unique value for each columns
concrete.nunique()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """

    


EDA ={"column ": concrete.columns,
      "mean": concrete.mean(),
      "median":concrete.median(),
      "mode":concrete.mode(),
      "standard deviation": concrete.std(),
      "variance":concrete.var(),
      "skewness":concrete.skew(),
      "kurtosis":concrete.kurt()}

EDA





# covariance for data set 
covariance = concrete.cov()
covariance

# Correlation matrix 
Correlation = concrete.corr()
Correlation
 


# according to correlation coefficient no correlation of  Administration & State with model_dffit
#According scatter plot strong correlation between model_dffit and rd_spend and 
#also some relation between model_dffit and m_spend.

#variance for each column
concrete.var()                   #rain column has low variance 




####### graphidf repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(concrete)

sns.heatmap(Correlation, annot=True, cmap='Blues')

sns.scatterplot(y="strength", x="cement", hue="water",size="age", data=concrete,  sizes=(50, 300))

#boxplot for every columns
concrete.columns
concrete.nunique()

#boxplot for every column

# Boxplot of independent variable distribution for each category of size_category


concrete.boxplot(column=['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',
       'fineagg', 'age', 'strength'])  



#normal

# Normalization function using z std. all are continuous data.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df = norm_func(concrete.iloc[:,0:8])
df.describe()


#final dataframe
model_df = pd.concat([concrete.iloc[:,[8]],df ], axis =1)


##################################

##################################
###upport Vector Machines MODEL###
"""5.	Model Building:
5.1	Perform Artificial Neural Network on the given datasets.
5.2	Use TensorFlow keras to build your model in Python and use Neural net package in R
5.3	Briefly explain the y in the documentation for each step in your own words.
5.4	Use different activation functions to get the best model.


"""


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
model.add(keras.layers.Dense(5000, activation='relu', input_dim=8))
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
model.fit(x=x_train,y=y_train,batch_size=50,epochs=100)


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

