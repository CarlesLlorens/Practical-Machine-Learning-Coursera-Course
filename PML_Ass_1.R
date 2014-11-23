##########################################################################################################
# Coursera Practical Machine Learning Course Assignment
# Version V3.0
# Date creation:          20/11/2014
# Date last modification: 23/11/2014
#
# Description:
#
#    Given a labeled dataset that conatins Human Activity Recognition features
#    and a unlabeled dataset for testing,
#    we make an exploratory data analysis and clean these datasets.
#    Later we partitioning the train dataset and construct a model using the random forest method 
#    to estimate features and evaluate the model using 4-fold cross validation. 
#    This method produces good accuracy.
#    Finally generate the prediction labels on test dataset and the corresponding output files.

# Steps :
#
# 1. Getting Data
#
#    The data is taken from the Human Activity Recognition programme at Groupware.
#    Web site for Train dataset: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
#    Web site for Test  dataset: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
#    These files are downloaded and save as csv file named pml-training.csv and pml-testing.csv
#    Train dataset contains 19622 observations and 160 variables
#    Test  dataset contains    20 observations and 160 variables
#
# 2. Exploratory Data Analisys
#
#    Examining the summary of the dataset reveals
#    that columns 1 to 6 do not provide relevant information.
#    So it is best to remove to simplify dataset. 
#    On the other hand, there are columns that contain Nan.
#    Delete also to simplify the data set.
#    As a result of Cleaning train dataset we have: 19622 observations and 54 variables
#    As a result of Cleaning test  dataset we have:    20 observations and 54 variables
#
# 3. Partitioning the training dataset into two parts (train subset and validation subset) 
#
#    Train subset has 60% of training dataset observations. This will be used to train the model.
#    Validation set has 60% of training dataset observations.This will be used for validation the model.
#
# 4. Constructing the model. 
#
#    I use the Random Forests method which applies bagging to tree learners. 
#    I evaluate the model using 4-fold cross validation. 
#    NOTE: Take a subset of 6000 observations in trainigdata because the entire 
#    training data causes Out of Memory in my computer.
#
# 5. Calculate prediction accuracy. 
#
#    Calculate the prediction accuracy of our model on the training data set.
#
# 6. Generate the answers and the output files.
#
#    I generate the labels resulting of predictions of test dataset over the model
#    and later generate the files needed by the assignment.
#
# IMPORTANT NOTE:
#
#    You must set-up the script according to your environment.
#    For this, you need set your working directory in line 80 of the script.
#
##########################################################################################################

# 0. Set-up and Initialization Environment
# ****************************************

# Packages needed:
# install.packages("caret")
# install.packages("randomForest")

library(caret)
library(randomForest)

# Clean up workspace
rm(list=ls())

# Set working directory
setwd("C:/Coursera/Practical_Machine_Learning")


# 1. Getting Data
# ***************

# Download data
# Create a function for download data from Internet and save in working directory
downloadDataset <- function(URL="", destFile="datapml.csv"){
  if(!file.exists(destFile)){
    download.file(URL, destFile, method="curl")
  }else{
    message("File already downloaded.")
  }
}

trainURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
downloadDataset(trainURL, "pml-training.csv")
downloadDataset(testURL , "pml-testing.csv")

train.original <- read.csv("pml-training.csv",na.strings=c("NA",""))
test.original  <- read.csv("pml-testing.csv", na.strings=c("NA",""))

# train.original contains 19622 observations and 160 variables
# test.original  contains    20 observations and 160 variables
cat(sprintf("Original datasets "))
cat(sprintf("train contains %d observations and %d variables", dim(train.original)[1],dim(train.original)[2]))
cat(sprintf("test  contains %d observations and %d variables", dim(test.original)[1],dim(test.original)[2]))


# 2. Exploratory Data Analisys
# ****************************

# Examining the summary of the dataset reveals
# that columns 1 to 6 do not provide relevant information.
# So it is best to remove to simplify dataset. 
# On the other hand, there are columns that contain Nan. 
# For example column 'kurtosis_yaw_belt'
# Delete also to simplify the data set.

train <- train.original[,-c(1:6)]
test  <- test.original[,-c(1:6)]

# Now, train contains 19622 observations and 154 variables
# and  test  contains    20 observations and 154 variables
cat(sprintf("train contains %d observations and %d variables", dim(train)[1],dim(train)[2]))
cat(sprintf("test  contains %d observations and %d variables", dim(test)[1], dim(test)[2]))


# Create a function to remove entire NA columns, 
# and apply this to train and test datasets.
# Next create a function that removes any variables 
# with missing NAs and apply this to train and test datasets.

rm.na.cols  <- function(x) { x[ , colSums( is.na(x) ) < nrow(x) ] }
train <- rm.na.cols(train)
test  <- rm.na.cols(test)

complete     <- function(x) {x[,sapply(x, function(y) !any(is.na(y)))] }
incomplete   <- function(x) {names( x[,sapply(x, function(y) any(is.na(y)))] ) }
train.na.var <- incomplete(train)
test.na.var  <- incomplete(test)
train <- complete(train)
test  <- complete(test)

# Now, train contains 19622 observations and 54 variables
# and  test  contains    20 observations and 54 variables
cat(sprintf("train contains %d observations and %d variables", dim(train)[1],dim(train)[2]))
cat(sprintf("test  contains %d observations and %d variables", dim(test)[1],dim(test)[2]))

# 3. Partitioning the training dataset into two parts. 
# ***************************************************
# First part will be used to train the model (60% of the data).
# Second part will be used as a validation set (40% of the data).

set.seed(123)
partition <- createDataPartition(y = train$classe, p = 0.6, list = FALSE)
traindata <- train[partition, ]
crossvalidationdata  <- train[-partition, ]


# 4. Constructing the model. 
# **************************

# I use the Random Forests method 
# which applies bagging to tree learners. 
# I evaluate the model using 4-fold cross validation. 
# NOTE: Take a subset of 6000 observations in trainigdata because the entire 
# training data causes Out of Memory in my computer.
time.start <-Sys.time()
trainIdx  <- sample(nrow(train), 6000)
traindata <-train[trainIdx,]
model <- train(classe ~ ., data = traindata, method = "rf", prox = TRUE, 
               tuneGrid=data.frame(mtry=3),
               trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))
model
time.end <-Sys.time()
time.taken <- time.end-time.start
cat(sprintf("Time taken for train the model: %05.3f minutes", time.taken))


# 5. Calculate prediction accuracy. 
# *********************************

# Calculate the prediction accuracy of our model on the training data set.
train_pred <- predict(model, traindata)
cmtr <- confusionMatrix(train_pred, traindata$classe); cmtr
cat(sprintf("Training Accuracy: %02.3f", cmtr$overall['Accuracy']))

# Calculate the prediction accuracy of our model on the cross-validation data set.
crossvalidation_pred <- predict(model, crossvalidationdata)
cmcv <- confusionMatrix(crossvalidation_pred, crossvalidationdata$classe); cmcv
cat(sprintf("Cross-Validation Accuracy: %02.3f", cmcv$overall['Accuracy']))

# 6. Generate the answers and the output files.
# *********************************************

#Generate the answers
answers <- predict(model, test)
answers <- as.character(answers)
answers

pml_write_files = function(x) {
  n = length(x)
  for (i in 1:n) {
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
                col.names = FALSE)
  }
}

pml_write_files(answers)
