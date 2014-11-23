
Practical Machine Learning
==========================

Assignment 1
============

Introduction
============

We have a dataset for the Human Activity Recognition from Groupware (see http://groupware.les.inf.puc-rio.br/har ) ,  in which exists collected data on 8 hours of activities of some healthy subjects.
This project uses a subset of this dataset, and attempts to use machine learning techniques to predict the classification of the activities according with the predictors stored in the dataset. Also, we see how good the predictor is and take metrics of errors.

Getting and Processing Data
===========================

Data was extracted from two csv files. The first was the training dataset, consisting of 19622 observations with classification labels. The second was the test dataset, consisting of 20 observations without classification labels.
The code use for download data and create datasets was:

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


Exploratory Data Analysis
=========================

Examining the summary of the train dataset, we observe that there are some columns that do not contain relevant information (columns 1 to 6). So it is best to remove to simplify dataset. 

    train <- train.original[,-c(1:6)]
    test  <- test.original[,-c(1:6)]

On the other hand, there are columns that contain Nan, for example column 'kurtosis_yaw_belt'. We decided delete it to simplify the data set.

    # Create a function to remove entire NA columns, 
    # and apply this to train and test datasets.
    # Next create a function that removes any variables 
    # with missing NAs and apply this to train and test datasets.

    rm.na.cols  <- function(x) { x[ , colSums( is.na(x) ) < nrow(x) ] }
    train <- rm.na.cols(train)
    test  <- rm.na.cols(test)

    complete    <- function(x) {x[,sapply(x, function(y) !any(is.na(y)))] }
    incompl     <- function(x) {names( x[,sapply(x, function(y) any(is.na(y)))] ) }
    train.na.var <- incompl(train)
    test.na.var  <- incompl(test)
    train <- complete(train)
    test  <- complete(test)

    # Now, train contains 19622 observations and 54 variables
    # and  test  contains    20 observations and 54 variables
    cat(sprintf("train contains %d observations and %d variables", dim(train)[1],dim(train)[2]))
    cat(sprintf("test  contains %d observations and %d variables", dim(test)[1],dim(test)[2]))



Machine Learning Model
======================

We use the Random Forest machine learning algorithm for this project.
Random Forest is one of the most accurate machine learning algorithms available. For a data set large enough produce a very accurate classifier.
For this algorithm, we decided partitioning the training dataset into two parts: the training part and de cros-validation part. The training part consists of 60% of the original training data and will be used to train the model. The cros-validation part consists of 40% of the original training part and will provide an estimate of how well the model is when applied on the test set.

    # Partitioning the training dataset into two parts. 
    # First part will be used to train the model (60% of the data).
    # Second part will be used as a validation set (40% of the data).

    set.seed(123)
    partition <- createDataPartition(y = train$classe, p = 0.6, list = FALSE)
    traindata <- train[partition, ]
    crossvalidationdata  <- train[-partition, ]

Next, we build the model and evaluate it using 4-fold cross validation. 
NOTE: We take a subset of 6000 observations in trainig data because the entire 
training data causes Out of Memory in my computer. We also control the processing time for do the train.

    # I use the Random Forests method 
    # which applies bagging to tree learners. 
    # I evaluate the model using 4-fold cross validation. 
    # Take a subset of 6000 observations in trainigdata because the entire 
    # training data causes Out of Memory in my computer.
    time.start <-Sys.time()
    trainIdx  <- sample(nrow(train), 6000)
    traindata <-train[trainIdx,]
    model <- train(classe ~ ., data = traindata, method = "rf", prox = TRUE, 
               tuneGrid=data.frame(mtry=3),
               trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))
    model
    time.end   <-Sys.time()
    time.taken <- time.end-time.start
    # Time needed for make the train.
    cat(sprintf("Time taken for train the model: %05.3f minutes", time.taken))

Prediction Accuracy and error metrics in datasets
=================================================

We calculate the accuracy and other metrics on the training set. 

    # Calculate the prediction accuracy of our model on the training data set.
    train_pred <- predict(model, traindata)
    cmtr <- confusionMatrix(train_pred, traindata$classe); cmtr
    cat(sprintf("Training Accuracy: %02.3f", cmtr$overall['Accuracy']))

The results are:

    Confusion Matrix and Statistics
    Reference
    Prediction    A    B    C    D    E
             A 1703    0    0    0    0
             B    0 1196    0    0    0
             C    0    0 1007    0    0
             D    0    0    0 1016    0
             E    0    0    0    0 1078
    
    Overall Statistics
               Accuracy : 1          
                 95% CI : (0.9994, 1)
    No Information Rate : 0.2838     
    P-Value [Acc > NIR] : < 2.2e-16  
    Kappa : 1          
    Mcnemar's Test P-Value : NA         
    
    Statistics by Class:
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    Prevalence             0.2838   0.1993   0.1678   0.1693   0.1797
    Detection Rate         0.2838   0.1993   0.1678   0.1693   0.1797
    Detection Prevalence   0.2838   0.1993   0.1678   0.1693   0.1797
    Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

The random forest model classifies all data correctly and give and accuracy rate of 100%.
We want to see how good the model is. It is necessary to predict our model with the cross-validation set obtained from the original dataset. 

    # Calculate the prediction accuracy of our model on the cross-validation dataset.
    crossvalidation_pred <- predict(model, crossvalidationdata)
    cmcv <- confusionMatrix(crossvalidation_pred, crossvalidationdata$classe); cmcv
    cat(sprintf("Cross-Validation Accuracy: %02.3f", cmcv$overall['Accuracy']))

The results of accuracy and other metrics are:

    Confusion Matrix and Statistics
    Reference
    Prediction    A    B    C    D    E
             A 2231   12    0    0    0
             B    1 1505   18    0    0
             C    0    1 1350   19    2
             D    0    0    0 1267   10
             E    0    0    0    0 1430
    Overall Statistics
                      
    Accuracy : 0.992           
    95% CI : (0.9897, 0.9938)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
    Kappa : 0.9898          
    Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            0.9996   0.9914   0.9868   0.9852   0.9917
    Specificity            0.9979   0.9970   0.9966   0.9985   1.0000
    Pos Pred Value         0.9947   0.9875   0.9840   0.9922   1.0000
    Neg Pred Value         0.9998   0.9979   0.9972   0.9971   0.9981
    Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    Detection Rate         0.2843   0.1918   0.1721   0.1615   0.1823
    Detection Prevalence   0.2859   0.1942   0.1749   0.1628   0.1823
    Balanced Accuracy      0.9987   0.9942   0.9917   0.9919   0.9958

The random forest model missclassifies 63 of 7846 observations and give and accuracy rate of 99.2%. Also, the sensitibity and specificity are near 1 in most cases. We think that will be a good predictor for the test dataset. 

Prediction Assignment
=====================
Finally, we apply the model over 20 test cases and generate the predicted labels in the test dataset. Later we generate the files needed in the assignment.

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
