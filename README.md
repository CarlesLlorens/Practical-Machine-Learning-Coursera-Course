

Practical Machine Learning
==========================

Asignment 1
===========

Introduction
============

First, sorry for my English. I’m Spanish.

We have a dataset for the Human Activity Recognition from Groupware (see http://groupware.les.inf.puc-rio.br/har ) ,  in which exists collected data on 8 hours of activities of some healthy subjects.
This project uses a subset of this dataset, and attempts to use machine learning techniques to predict the classification of the activities according with the predictors stored in the dataset. Also, we see how good the predictor is and take metrics of errors.

Getting and Processing Data
===========================

Data was extracted from two csv files. The first was the training dataset, consisting of 19622 observations with classification labels. The second was the test dataset, consisting of 20 observations without classification labels.
Exploratory Data Analysis
Examining the summary of the train dataset, we observe that there are some columns that do not contain relevant information (columns 1 to 6). So it is best to remove to simplify dataset. 
On the other hand, there are columns that contain Nan, for example column 'kurtosis_yaw_belt'. We decided delete it to simplify the data set.

Machine Learning Model
======================
We use the Random Forest machine learning algorithm for this project.
Random Forest is one of the most accurate machine learning algorithms available. For a data set large enough produce a very accurate classifier.
For this algorithm, we decided partitioning the training dataset into two parts: the training part and de test part. The training part consists of 60% of the original training data and will be used to train the model. The test part consists of 40% of the original training part and will provide an out of sample estimate of how well the model will perform on the test set.
Next, we build the model using 4-fold cross validation. 
Take a subset of 6000 observations in trainigdata because the entire 
training data causes Out of Memory in my computer.

Accuracy In Test Dataset
========================
We calculate the accuracy on the training set. The results are:

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
We want to see how good the model is. It is necessary to predict our model with the cross-validation  set obtained from the original dataset. The results are:

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

The random forest model missclassifies 63 of 7846 observations and give and accuracy rate of 99.2%..

Prediction Assignment
=====================
We apply the machine learning algorithm over 20 test cases in the testing data set provided.

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
