# PML_CourseProject
JPM  
February 17, 2016  
####Introduction  
One thing that people regularly do is quantify *how much* of a particular activity they do, but they rarely quantify *how well* they do it. In this project, the goal was to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to make predictions about how they performed a weight-lifting exercise. The participants performed the lift correctly, as well as four different incorrect executions of the lift.

####Online Versions
Online versions of this report can be found at the below links:

[https://github.com/jmigaleddi/practicalmachinelearning](https://github.com/jmigaleddi/practicalmachinelearning)
[http://jmigaleddi.github.io/practicalmachinelearning/](http://jmigaleddi.github.io/practicalmachinelearning/)

####Executive Summary
This report provides an analysis of the data collected during a weight-lifting experiment. The goal was to use the collected data to develop a model to predict how the lift was performed: either correctly, or one of four different incorrect lifts. Upon arriving at a final specification, the model would be tested for accuracy against 20 independent observations.

More information on the data used for this analysis can be found  [here.](http://groupware.les.inf.puc-rio.br/har)

After some initial preparation, the data was analyzed by generating different models and, using k-fold cross-validation, checking their estimates for out-of-sample accuracy. The three model types tested were:  

1. Classification Tree (`method = "rpart"`)  
2. Random Forest (`method = "rf"`)  
3. Bagged Classification Tree (`method = "treebag"`)  

The results indicated that the random forest model provided the best model metrics, and was the one chosen to predict the test set. Accuracy on the test set was 100% (20 correct out of 20).

####Model Creation and Specification  
#####Reading in the data

```r
##Load the required packages
library(plyr); library(dplyr); library(ggplot2); library(caret); library(e1071)

##Read in the training and testing data
train <- tbl_df(read.csv("./data/pml-training.csv"))
test <- tbl_df(read.csv("./data/pml-testing.csv"))

##Remove all of the summary observations
train <- train[train$new_window =="no",]
```

The training data set was quite extensive, with over 150 measurements and derived measurements collected from the experiment. A visual inspection of the test data showed that only a small subset of those variables (about 50) were going to be available to be used for the prediction exercise, so the rest were removed from the training set.

#####Dimension Reduction

```r
##Keep Only the columns that are available to be used in the prediction exercise
train.match <- train[, c(8:11, 37:49, 60:68, 84:86, 102, 113:124, 140, 151:160)]
test.match <- test[, c(8:11, 37:49, 60:68, 84:86, 102, 113:124, 140, 151:160)]

matrix(data = rbind(dim(train.match), dim(test.match)),
       nrow = 2,
       dimnames = list(c("Training", "Test"), c("Observations", "Variables")))
```

```
##          Observations Variables
## Training        19216        53
## Test               20        53
```

Checking the dimensions of the newly filtered data set showed that the data had been reduced down to about 50 variables to be used for the prediction exercise.

In the interest of further data reduction, the remaining variables were checked for high levels of collinearity.


```r
##Create a table showing the highly correlated predictors
corrMat <- cor(train.match[,-53])
diag(corrMat) <- 0
which(corrMat > 0.9, arr.ind = T)
```

```
##                  row col
## total_accel_belt   4   1
## accel_belt_y       9   1
## roll_belt          1   4
## accel_belt_y       9   4
## roll_belt          1   9
## total_accel_belt   4   9
## gyros_forearm_z   46  33
## gyros_dumbbell_z  33  46
```

```r
##Remove the highly correlated predictors
highCorPred <- findCorrelation(corrMat, cutoff = 0.90)
train.match.corfilter <- train.match[,-highCorPred]
```

#####Model Tuning/Parameters
Upon sufficiently preparing the training data set, model creation and testing began. Before building any models, some model tuning/parameters were specified. k-fold cross-validation was implemented via the `trainControl` function to derive an estimate of out-of-sample accuracy. 3 folds were used with no repeats given the robustness of the data set.

An object was also created that was passed to the `tuneGrid` argument in the random forest model to limit the number of variables to be tested at each split. This was mainly done to achieve processing efficiency. 


```r
train_control <- trainControl(method = "cv", number = 3)
rf.tunegrid <- data.frame(mtry = c(5))
```

#####Model Creation
Three different model methods were utilized: classification tree, random forest, and bagged classification tree. Each were checked for estimated out-of-sample accuracy using cross-validation.


```r
Fit.rpart <- train(train.match$classe ~ ., 
                   method = "rpart", 
                   trControl = train_control,
                   data = train.match.corfilter)

Fit.rf <- train(train.match$classe ~ ., 
                method = "rf", 
                trControl = train_control,
                tuneGrid = rf.tunegrid,
                data = train.match.corfilter)

Fit.treebag <- train(train.match$classe ~ ., 
                     method = "treebag", 
                     trControl = train_control,
                     data = train.match.corfilter)
```

Comparing the results of the three models showed that the random forest and bagged classification tree were clearly better models. The random forest correctly predicted all of the training data, while the bagged tree missed only seven of the more than 19,000 predictions. The out-of-sample error estimates from the cross-validation were both more than 98%. Given that the random forest model performed slightly better, that model was chosen for the prediction exercise.


```r
##Create the predictions on the training set
rpart.pred <- predict(Fit.rpart, train.match)
rf.pred <- predict(Fit.rf, train.match)
treebag.pred <- predict(Fit.treebag, train.match)

##Create a table comparing in-sample and out-of-sample accuracy
acc.table <- cbind(rbind(confusionMatrix(train.match$classe, rpart.pred)$overall["Accuracy"],
                         confusionMatrix(train.match$classe, rf.pred)$overall["Accuracy"],
                         confusionMatrix(train.match$classe, treebag.pred)$overall["Accuracy"]),
                   rbind(Fit.rpart$results[1,2],
                         Fit.rf$results[2],
                         Fit.treebag$results[2]))
row.names(acc.table) <- c("Class. Tree", "Random Forest", "Bagged Class. Tree")
colnames(acc.table) <- c("In-Sample Accuracy", "Est. Out-of-Sample Accuracy")

acc.table
```

```
##                    In-Sample Accuracy Est. Out-of-Sample Accuracy
## Class. Tree                 0.4751249                   0.5021875
## Random Forest               1.0000000                   0.9933908
## Bagged Class. Tree          0.9997918                   0.9858450
```

####Prediction Results

The application of the random forest model to the test set yielded perfect results (20 correct out of 20). Comparison of the performance of the random forest model and the bagged classification tree on the prediction quiz actually yielded identically perfect results, despite the small differences in the cross-validated accuracy.


```r
all.equal(predict(Fit.rf, test.match), predict(Fit.treebag, test.match))
```

```
## [1] TRUE
```
