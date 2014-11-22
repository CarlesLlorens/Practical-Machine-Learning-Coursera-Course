

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
Next, proceed with fit the random forest model and K-Fold Cross Validation with K=4.

Accuracy In Test Dataset
========================
We calculate the accuracy on the training set.

The random forest model classifies all data correctly and give and accuracy rate of 100%.
We want to see how good the model is. It is necessary to predict our model with the cross-validation  set obtained from the original dataset.

The random forest model missclassifies 63 of 7846 observations and give and accuracy rate of 99.2%..

Prediction Assignment
=====================
We apply the machine learning algorithm over 20 test cases in the testing data set provided.


