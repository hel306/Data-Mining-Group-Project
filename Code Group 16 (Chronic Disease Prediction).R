library(xgboost)
library(caret)
library(rpart)
library(neuralnet)
library(pROC)
library(randomForest)
library(e1071)
library(dplyr)
data <- read.csv("C:/Users/mintmachine/OneDrive/health.csv")

data$ChronicDisease <- as.factor(data$ChronicDisease)
data$Sex <- as.factor(data$Sex)
data$PhysicalActivities <- as.factor(data$PhysicalActivities)
data$SmokerStatus <- as.factor(data$SmokerStatus)
data$AlcoholDrinkers <- as.factor(data$AlcoholDrinkers)
data$CovidPos <- as.factor(data$CovidPos)
data$AgeCategory <- as.factor(data$AgeCategory)

data[is.na(data)] <- 0

set.seed(110)
trainIndex <- createDataPartition(data$ChronicDisease, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

#LRM
log_model <- glm(ChronicDisease ~    Sex + PhysicalActivities + SleepHours + BMI + 
                AlcoholDrinkers + SmokerStatus + AgeCategory + CovidPos,
             data = trainData, family = binomial)
summary(log_model)
log_prob <- predict(log_model, testData, type = "response")
log_roc <- roc(testData$ChronicDisease, log_prob)
log_pred <- ifelse(log_prob > 0.5, "Yes", "No")
log_pred <- factor(log_pred, levels = levels(testData$ChronicDisease))
confusionMatrix(log_pred, testData$ChronicDisease)

#DTM
tree_model <- rpart(ChronicDisease ~    Sex + PhysicalActivities + SleepHours + BMI + 
                       AlcoholDrinkers + SmokerStatus + AgeCategory + CovidPos,
                    data = trainData, method = "class")
tree_model <- rpart(ChronicDisease ~ ., data = trainData, method = "class",
                    parms = list(prior = c(0.5, 0.5)))

tree_prob <- predict(tree_model, testData, type = "prob")[,2]


prob <- predict(tree_model, testData, type = "prob")
tree_roc <- roc(testData$ChronicDisease, tree_prob)
tree_pred <- predict(tree_model, testData, type = "class")
confusionMatrix(tree_pred, testData$ChronicDisease)

#RFM
rf_model <- randomForest(ChronicDisease ~    Sex + PhysicalActivities + SleepHours + BMI + 
                            AlcoholDrinkers + SmokerStatus + AgeCategory + CovidPos,
                         data = trainData, ntree = 100)

rf_prob <- predict(rf_model, testData, type = "prob")[,2]
rf_roc <- roc(testData$ChronicDisease, rf_prob)
rf_pred <- predict(rf_model, testData)
confusionMatrix(rf_pred, testData$ChronicDisease)

#Sampling Data
set.seed(100)
sampledData <- trainData[sample(1:nrow(trainData), 10000), ]

#SVM
svm_model <- svm(ChronicDisease ~    Sex + PhysicalActivities + SleepHours + BMI + 
                    AlcoholDrinkers + SmokerStatus + AgeCategory + CovidPos,
                 data = sampledData, kernel = "radial" , class.weights = c("Yes" = 0.5, "No" = 0.5),
                 probability = TRUE)

svm_prob <- attr(predict(svm_model, testData, probability = TRUE), "probabilities")[,2]
svm_roc <- roc(testData$ChronicDisease, svm_prob)
summary(svm_prob)
svm_pred <- predict(svm_model, testData)
confusionMatrix(svm_pred, testData$ChronicDisease)

#KNN
numeric_columns <- sapply(data, is.numeric)
data[numeric_columns] <- scale(data[numeric_columns])
knn_model <- train(ChronicDisease ~    Sex + PhysicalActivities + SleepHours + BMI + 
                      AlcoholDrinkers + SmokerStatus + AgeCategory + CovidPos,
                   data = sampledData, method = "knn", trControl = trainControl(method = "cv", number = 10))
summary(knn_model)
knn_prob <- predict(knn_model, testData, type = "prob")[,2]
knn_roc <- roc(testData$ChronicDisease, knn_prob)
knn_pred <- predict(knn_model, testData) 
confusionMatrix(knn_pred, testData$ChronicDisease)

#Naive Bayes
trainData$ChronicDisease <- as.factor(trainData$ChronicDisease)
testData$ChronicDisease <- as.factor(testData$ChronicDisease)
nb_model <- naiveBayes(ChronicDisease ~ ., data = trainData)
summary(nb_model)
nb_predictions <- predict(nb_model, testData, type = "raw")
nb_pred_class <- predict(nb_model, testData)    
nb_roc <- roc(testData$ChronicDisease, nb_predictions[,2])
confusionMatrix(nb_pred_class, testData$ChronicDisease)

plot(log_roc, col = "blue", print.auc = FALSE, main = "ROC Curve Comparison")
plot(tree_roc, col = "green", add = TRUE, print.auc = FALSE)
plot(rf_roc, col = "red", add = TRUE, print.auc = FALSE)
plot(svm_roc, col = "purple", add = TRUE, print.auc = FALSE)
plot(knn_roc, col = "orange", add = TRUE, print.auc = FALSE)
plot(nb_roc, col = "brown", add = TRUE, print.auc = FALSE)

legend("bottomright", 
       legend = c(paste("Logistic Regression AUC=", round(auc(log_roc), 2)),
                  paste("Decision Tree AUC=", round(auc(tree_roc), 2)),
                  paste("Random Forest AUC=", round(auc(rf_roc), 2)),
                  paste("SVM AUC=", round(auc(svm_roc), 2)),
                  paste("KNN AUC=", round(auc(knn_roc), 2)),
                  paste("Naive Bayes AUC=", round(auc(nb_roc), 2))),
       col = c("blue", "green", "red", "purple","orange","brown"),
       lwd = 3,
       cex = 0.6)