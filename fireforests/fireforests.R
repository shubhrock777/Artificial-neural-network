
forest <- read.csv(file.choose(), stringsAsFactors = TRUE)

summary(forest)

# Partition Data into train and test data
forest_train <- forest[1:417, ]
forest_test  <- forest[418:517, ]

colnames(forest_train)
# Training a model on the data ----

## Training a model on the data ----
# train the neuralnet model
library(neuralnet)

# simple ANN with only a single hidden neuron
concrete_model <- neuralnet(formula = area ~ FFMC+DMC+DC+ISI+temp+RH+wind+rain+dayfri+
                              daymon+daysat+daysun+daythu+daytue+daywed+monthapr+
                              month+aug+monthdec+monthfab+monthjan+monthjul+monthjun+
                              monthmar+monthmay+monthnov+monthoct+monthsep,
                              data = forest_train )


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
concrete_model2 <- neuralnet( area ~ FFMC+DMC+DC+ISI+temp+RH+wind+rain+dayfri+
                                daymon+daysat+daysun+daythu+daytue+daywed+monthapr+
                                month+aug+monthdec+monthfab+monthjan+monthjul+monthjun+
                                monthmar+monthmay+monthnov+monthoct+monthsep,
                              data = forest_train, hidden = 5)


# plot the network
plot(concrete_model2)

# evaluate the results as we did before
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)