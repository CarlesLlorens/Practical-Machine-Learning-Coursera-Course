Prac_Mach_Learning
==================

Practical Machine Learning

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
