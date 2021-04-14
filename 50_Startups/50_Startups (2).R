#######################Q1
# Loading the data into a variable


startup_50 <- read.csv(file.choose())

# Getting Summary of data
summary(startup_50)



# Variance


var(startup_50$Administration)



var(startup_50$Profit)



sd(startup_50$Administration)


sd(startup_50$Profit)

#Checking how many city are in state
unique(startup_50$State)



#startup_50 <- cbind(startup_50,ifelse(startup_50$State=="New York",1,0), ifelse(startup_50$State=="California",1,0),  ifelse(startup_50$State=="Florida",1,0))


# Renaming the column3
#colnames(startup_50)[6] <- "New York"
#colnames(startup_50)[7] <- "California"
#colnames(startup_50)[8] <- "Florida"     


colnames(startup_50)

#Creating Model

# Partition Data into train and test data
startup_50_train <- startup_50[1:40, ]
startup_50_test  <- startup_50[41:50, ]

colnames(startup_50)
# Training a model on the data ----

## Training a model on the data ----
# train the neuralnet model
library(neuralnet)

# simple ANN with only a single hidden neuron
concrete_model <- neuralnet(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend+State,
                            data = startup_50_train)


# visualize the network topology
plot(concrete_model)

## Evaluating model performance 

# obtain model results
# results_model <- NULL

results_model <- compute(concrete_model, concrete_test[1:8])
# obtain predicted strength values
str(results_model)
predicted_strength <- results_model$net.result

# examine the correlation between predicted and actual values
cor(predicted_strength, concrete_test$strength)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic + 
                               coarseagg + fineagg + age,
                             data = concrete_train, hidden = 5)


# plot the network
plot(concrete_model2)

# evaluate the results as we did before
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)