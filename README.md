# expediteOQ
install.packages("caret")
install.packages("readxl")
install.packages("corrplot")
install.packages("tidyverse")
install.packages("e1071")
install.packages("neuralnet")
install.packages("car")
library(caret) # For hyperparameter tuning
library(readxl)
library(ISLR)
library(ggplot2)
library(corrplot)
library(tidyverse)
library(leaps)
library(e1071) # For SVR
library(randomForest) # For RF
library(gbm) # For GBM
library(neuralnet) # For ANN
library(car)

#############  DATA PREPROCESSING   ##################
setwd("/Users/venkatasurendrapeddireddi/Documents")
getwd()
# Read the historic data from the Excel file
data <- read_excel("P396OQ.xlsx")

# Convert 'orientation' into a factor
data$orientation <- factor(data$orientation)
# Convert the factor levels into numerical values using as.numeric()
data$orientation <- as.numeric(data$orientation)

# Convert 'ID' into a factor
data$ID <- factor(data$ID)
# Convert the factor levels into numerical values using as.numeric()
data$ID <- as.numeric(data$ID)

#checking for correlation
# Calculate the correlation matrix
cor(data)
cor_matrix <- cor(data)
# Plot the correlation matrix with colors
corrplot(cor_matrix,  method = "pie", tl.cex = 1.2, cl.cex = 1.2)
head(data)

#bulkdensity,pourabilityphase,transformation have high correlation between them & ID, orientation also have high correlatiion bn them
# Remove unnecessary columns
data <- data[, !names(data) %in% c("powderbulkdensity", "pourability", "ID","phasetransformation")]

#verifying how my data looks like
head(data)
#finding correlation between variables
cor(data)
# Calculate the correlation matrix
cor_matrix <- cor(data)
# Plot the correlation matrix with colors
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.cex = 1.2, cl.cex = 1.2)
head(data)

#Checking for collinearity
#collinearity
vif_values <- vif(lm(E ~ ., data = data))
vif_values

#Checking for non linearity of data
# Fit the linear regression model
lm_model <- lm(E ~ ., data = data)
# Calculate the residuals
residuals <- resid(lm_model)
# Create the residual vs. E plot
plot(data$E, residuals, pch = 16, col = "blue",
     xlab = "E (Actual values)", ylab = "Residuals",
     main = "Residuals vs. E (Linear Regression)")
abline(h = 0, lty = 2)

# Number of rows in the dataset
total_rows <- nrow(data)
total_rows
# Number of rows for training data (80%)
train_rows <- round(0.8 * total_rows)
train_rows
# Randomly select row indices for training data
train_indices <- sample(1:total_rows, train_rows)
train_indices
# Split the data into training and test sets
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]
print(test_data)
#verifying boxplots to find outliers in the data
boxplot(train_data$E)

################ MODEL BUILDING ##################
#1 LINEAR REGRESSION
# Train the Linear Regression model with k-fold CV
set.seed(17)
lm_model <- train(E ~ ., data = train_data, method = "lm", trControl = trainControl(method = "cv", number = 10))
# Get the best Linear Regression model
lm_best_model <- lm_model$finalModel
# Make predictions on the test data
lm_predictions <- predict(lm_best_model, newdata = test_data)
# Calculate RMSE for Linear Regression
lm_rmse <- sqrt(mean((test_data$E - lm_predictions)^2))
lm_rmse


#2 SVR
set.seed(17)
num_folds <- 10
fold_indices <- createFolds(train_data$E, k = num_folds) 
control <- trainControl(method = "cv", number = num_folds)
# Specify the tuning parameter grid for SVM 
tune_grid <- expand.grid(sigma = c(0.1, 1, 10),C = c(0.1, 1, 10))
# Train SVM model with hyperparameter tuning
svm_model <- train(E ~ ., data = train_data, method = "svmRadial",
                   trControl = control, tuneGrid = tune_grid, metric = "RMSE", verbose = FALSE)
# Print best hyperparameters
cat("Best Hyperparameters for SVM:\n") 
cat("sigma:", svm_model$bestTune$sigma, "\n") 
cat("C:", svm_model$bestTune$C, "\n")
# Train final SVM model with best hyperparameters on full training data 
final_svm_model <- svm(formula = E ~ ., data = train_data,kernel = "radial",scale = svm_model$bestTune$sigma, cost = svm_model$bestTune$C)
# Use the trained SVM model for prediction on test data 
preds_svm <- predict(final_svm_model, newdata = test_data)
# Calculate RMSE for SVM model on test data
svm_rmse<- sqrt(mean((test_data$E - preds_svm)^2)) 
svm_rmse
# Print prediction results for SVM model 
cat("Prediction results on test data (SVM):\n") 
print(preds_svm)

#3 RANDOM FOREST
# Random Forest
set.seed(17)
# Hyper parameter tuning using CV
Red_reg = trainControl(method = "cv" , number = 10) 
tune.out=tune(randomForest,E ~ .,data=train_data, importance = TRUE, trControl = Red_reg,
                                                                  ranges=list(ntree=c(10, 20,50), mtry=c(2,3,4)))
summary(tune.out)
bestmod = tune.out$best.model
summary(bestmod)
bestmod$ntree
bestmod$mtry
yhat_preds_rf <- predict(bestmod, newdata = test_data)
# Calculate RMSE for Random Forest
rf_rmse <- sqrt(mean((test_data$E - yhat_preds_rf)^2))
rf_rmse

#4 GBM
# GBM with hyperparameter tuning and k-fold CV
set.seed(17)
tuned_gbm <- train(E ~ ., data = train_data, method = "gbm",
                   tuneGrid = expand.grid(n.trees = c(10, 20), 
                                          interaction.depth = c(1, 2), 
                                          shrinkage = c(0.01, 0.1),
                                          n.minobsinnode = c(10, 20)),
                   trControl = trainControl(method = "cv", number = 10, verboseIter = TRUE))
# Get the best GBM model
gbm_best_model <- tuned_gbm$finalModel
# Make predictions on the test data
gbm_predictions <- predict(gbm_best_model, newdata = test_data) 
# Calculate RMSE for GBM
gbm_rmse <- sqrt(mean((test_data$E - gbm_predictions)^2))
gbm_rmse

#5 ANN
# ANN with hyperparameter tuning and k-fold CV
set.seed(17)
tuned_ann <- train(E ~ ., data = train_data, method = "nnet",
                   tuneGrid = data.frame(size = c(1, 2, 3), decay = c(0.1, 0.01,0.001)),
                   trControl = trainControl(method = "cv", number = 10))
# Get the best ANN model
ann_best_model <- tuned_ann$finalModel
# Make predictions on the test data
ann_predictions <- predict(ann_best_model, newdata = test_data)
# Calculate RMSE for ANN
ann_rmse <- sqrt(mean((test_data$E - ann_predictions)^2))
ann_rmse


############ PEDICTION ####################
# Read the data from the 'P396pred' Excel file
prediction_data <- read_excel("P396OQpred.xlsx")
head(prediction_data)

# Convert 'orientation' into a factor
prediction_data$orientation <- factor(prediction_data$orientation)
# Convert the factor levels into numerical values using as.numeric()
prediction_data$orientation <- as.numeric(prediction_data$orientation)

# Convert 'ID' into a factor
prediction_data$ID <- factor(prediction_data$ID)
# Convert the factor levels into numerical values using as.numeric()
prediction_data$ID <- as.numeric(prediction_data$ID)

# Remove unnecessary columns
pred_data <- prediction_data[, !names(prediction_data) %in% c("bulkdensity", "pourability", "ID","phasetransformation")]
head(pred_data)

# Make predictions using the linear regression model
lm_pred <- predict(lm_model, newdata = pred_data)

# Add the predicted E values to the prediction data
pred_data$E <- lm_pred

# View the predictions
pred_data
write.csv(pred_data, file = "predictions.csv", row.names = FALSE)
