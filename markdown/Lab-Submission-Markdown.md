Business Intelligence Lab 5
================
Chicken-nuggets
17th october 2023

- [Student Details](#student-details)

# Student Details

<table style="width:90%;">
<colgroup>
<col style="width: 45%" />
<col style="width: 44%" />
</colgroup>
<tbody>
<tr class="odd">
<td><strong>Student ID Number and Name</strong></td>
<td><ol type="1">
<li>137118 Fatoumata Camara</li>
<li>127039 Ayan Ahmed</li>
<li>136869 Birkanwal Bhambra</li>
<li>127602 Trevor Anjere</li>
<li>133824 Habiba Siba</li>
</ol></td>
</tr>
<tr class="even">
<td><strong>BBIT 4.2 Group</strong></td>
<td>Chicken-nuggets</td>
</tr>
</tbody>
</table>

``` r
if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: languageserver

``` r
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: mlbench

``` r
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: caret

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: klaR

    ## Loading required package: MASS

``` r
## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: e1071

``` r
## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: readr

``` r
## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: LiblineaR

``` r
## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: naivebayes

    ## naivebayes 0.9.7 loaded

``` r
data("PimaIndiansDiabetes")
##Naive Bayes----

##Spillting the Dataset
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,  
                                   p = 0.70, list = FALSE)
PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]

##Training a Naive Bayes classifier
## a. using "NaiveBayes()" function in the "klaR" package 
PimaIndiansDiabetes_model_nb_klaR <-
  klaR::NaiveBayes(diabetes ~ .,
                   data = PimaIndiansDiabetes_train)


##b. using "naiveBayes()" function in the e1071 package
PimaIndiansDiabetes_model_nb_e1071 <-
  klaR::NaiveBayes(diabetes ~ .,
                   data = PimaIndiansDiabetes_train)


## Test the trained Naive Bayes model
PimaIndiansDiabetes_model_nb_e1071 <- naiveBayes(diabetes ~ ., data = PimaIndiansDiabetes_train, laplace = 1)

predictions_nb_e1071 <-
  predict(PimaIndiansDiabetes_model_nb_e1071,
          PimaIndiansDiabetes_test[, 1:9])

## results
print(PimaIndiansDiabetes_model_nb_e1071)
```

    ## 
    ## Naive Bayes Classifier for Discrete Predictors
    ## 
    ## Call:
    ## naiveBayes.default(x = X, y = Y, laplace = laplace)
    ## 
    ## A-priori probabilities:
    ## Y
    ##       neg       pos 
    ## 0.6505576 0.3494424 
    ## 
    ## Conditional probabilities:
    ##      pregnant
    ## Y         [,1]     [,2]
    ##   neg 3.240000 2.969684
    ##   pos 4.957447 3.747127
    ## 
    ##      glucose
    ## Y         [,1]     [,2]
    ##   neg 109.0629 26.10013
    ##   pos 141.2181 31.99232
    ## 
    ##      pressure
    ## Y         [,1]     [,2]
    ##   neg 68.38000 17.11402
    ##   pos 71.40426 22.14841
    ## 
    ##      triceps
    ## Y         [,1]     [,2]
    ##   neg 20.33714 14.78307
    ##   pos 21.08511 17.81195
    ## 
    ##      insulin
    ## Y        [,1]      [,2]
    ##   neg  70.360  89.33092
    ##   pos 100.633 141.31244
    ## 
    ##      mass
    ## Y         [,1]     [,2]
    ##   neg 30.63229 7.253827
    ##   pos 34.86649 6.816902
    ## 
    ##      pedigree
    ## Y          [,1]      [,2]
    ##   neg 0.4251314 0.2696929
    ##   pos 0.5288404 0.3627678
    ## 
    ##      age
    ## Y         [,1]     [,2]
    ##   neg 30.99429 10.96543
    ##   pos 37.61170 10.99311

``` r
caret::confusionMatrix(predictions_nb_e1071,
                       PimaIndiansDiabetes_test$diabetes)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg 117  30
    ##        pos  33  50
    ##                                           
    ##                Accuracy : 0.7261          
    ##                  95% CI : (0.6636, 0.7826)
    ##     No Information Rate : 0.6522          
    ##     P-Value [Acc > NIR] : 0.01019         
    ##                                           
    ##                   Kappa : 0.4015          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.80106         
    ##                                           
    ##             Sensitivity : 0.7800          
    ##             Specificity : 0.6250          
    ##          Pos Pred Value : 0.7959          
    ##          Neg Pred Value : 0.6024          
    ##              Prevalence : 0.6522          
    ##          Detection Rate : 0.5087          
    ##    Detection Prevalence : 0.6391          
    ##       Balanced Accuracy : 0.7025          
    ##                                           
    ##        'Positive' Class : neg             
    ## 

``` r
##confusion matrix
plot(table(predictions_nb_e1071,
           PimaIndiansDiabetes_test$diabetes))
```

![](Lab-Submission-Markdown_files/figure-gfm/Using%20NaiveBayes-1.png)<!-- -->

``` r
# Load the Pima Indians Diabetes dataset
data("PimaIndiansDiabetes")

# Split the dataset into training and testing sets
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes, p = 0.65, list = FALSE)
PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]

# Train a logistic regression model
train_control <- trainControl(method = "boot", number = 500)

PimaIndiansDiabetes_model_lm <- caret::train(
  diabetes ~ pregnant + glucose + pressure + triceps + insulin + mass + pedigree + age,
  data = PimaIndiansDiabetes_train,
  trControl = train_control,
  na.action = na.omit,
  method = "glm",
  family = binomial(),
  metric = "Accuracy"
)

## test model
predictions_lm <- predict(PimaIndiansDiabetes_model_lm,
                          PimaIndiansDiabetes_test[, 1:9])

## View the Accuracy and the predicted values 
print(PimaIndiansDiabetes_model_lm)
```

    ## Generalized Linear Model 
    ## 
    ## 500 samples
    ##   8 predictor
    ##   2 classes: 'neg', 'pos' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (500 reps) 
    ## Summary of sample sizes: 500, 500, 500, 500, 500, 500, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.7702677  0.4755469

``` r
print(predictions_lm)
```

    ##   [1] pos neg neg neg pos neg pos neg pos neg pos pos neg neg neg neg neg neg
    ##  [19] neg pos neg neg neg neg neg neg neg neg neg neg neg pos neg neg neg pos
    ##  [37] neg neg neg pos neg neg pos neg neg neg neg neg pos neg pos pos pos neg
    ##  [55] neg neg neg neg pos neg pos neg neg neg neg neg neg pos neg pos neg neg
    ##  [73] neg neg pos neg pos pos neg pos neg pos neg pos neg neg neg pos neg neg
    ##  [91] pos neg neg pos pos neg neg neg pos pos neg neg pos neg pos neg neg pos
    ## [109] neg neg neg pos neg pos pos neg neg neg neg neg neg neg neg neg neg pos
    ## [127] pos pos pos neg neg neg pos neg neg neg neg neg pos neg neg neg neg neg
    ## [145] pos neg neg pos neg pos neg neg neg neg neg neg neg neg neg pos pos pos
    ## [163] neg neg pos pos neg neg neg neg neg neg pos neg neg neg pos neg neg neg
    ## [181] neg neg neg neg pos neg neg neg neg neg pos neg neg neg pos neg neg neg
    ## [199] neg neg pos neg neg neg neg neg neg neg pos neg neg pos neg neg pos neg
    ## [217] neg neg neg neg neg neg pos neg pos neg pos neg pos neg neg pos neg neg
    ## [235] neg neg neg neg pos pos pos neg pos neg neg neg neg neg neg neg neg neg
    ## [253] neg neg neg neg pos neg neg pos neg neg neg pos neg pos neg neg
    ## Levels: neg pos

``` r
## Use the model to make a prediction on unseen new data
new_data <-
  data.frame( pregnant = c(4), 
              glucose = c(160),
              pressure = c(149),
              triceps = c(50),
              insulin = c(450),
              mass = c(50),
              pedigree = c(1.9),
             age = c(48), check.names = FALSE)


# We now use the model to predict the output based on the unseen new data:
predictions_lm_new_data <-
  predict(PimaIndiansDiabetes_model_lm, new_data)

# The output below refers to the total orders:
print(predictions_lm_new_data)
```

    ## [1] pos
    ## Levels: neg pos

``` r
data("PimaIndiansDiabetes")
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes,
                                   p = 0.60, list = FALSE)
PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]

##regression----
## a. using 10-fold cross validation
train_control <- trainControl(method = "cv", number = 10)

PimaIndiansDiabetes_model_lm <-
  caret::train(diabetes ~ .,
               data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "glm", metric = "Accuracy")


## b. Testing the trained linear model
predictions_lm <- predict(PimaIndiansDiabetes_model_lm, PimaIndiansDiabetes_test[, -9])

## View the Accuracy and the predicted values
print(PimaIndiansDiabetes_model_lm)
```

    ## Generalized Linear Model 
    ## 
    ## 461 samples
    ##   8 predictor
    ##   2 classes: 'neg', 'pos' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 415, 415, 415, 415, 414, 415, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.7528215  0.4200633

``` r
print(predictions_lm)
```

    ##   [1] neg pos neg pos neg neg pos pos neg neg pos pos neg pos pos pos pos pos
    ##  [19] neg neg neg neg neg pos neg neg neg neg neg neg pos neg neg neg neg neg
    ##  [37] neg pos pos neg neg neg pos neg neg neg pos pos neg neg neg neg neg neg
    ##  [55] neg pos pos neg neg pos neg neg neg neg pos pos pos neg neg neg neg pos
    ##  [73] neg neg neg pos neg neg neg neg neg pos neg neg pos neg neg pos neg pos
    ##  [91] pos neg pos neg pos pos neg neg neg pos neg neg neg neg neg neg pos pos
    ## [109] pos pos neg pos neg neg pos pos neg neg pos pos pos neg pos neg neg neg
    ## [127] pos neg neg neg neg neg neg pos pos neg neg pos pos neg neg neg neg pos
    ## [145] neg pos pos neg neg neg neg neg neg neg neg pos pos neg pos neg neg neg
    ## [163] pos neg neg neg pos neg neg neg neg pos pos neg neg neg pos neg pos neg
    ## [181] neg neg pos pos neg pos neg neg neg neg neg pos neg neg neg neg neg pos
    ## [199] neg pos neg neg neg neg neg neg neg pos neg neg neg neg neg neg neg neg
    ## [217] neg neg neg neg neg pos neg pos neg neg pos neg neg pos neg neg neg neg
    ## [235] neg neg neg pos neg pos pos neg pos neg neg neg pos neg neg neg neg neg
    ## [253] neg pos pos neg neg neg neg neg neg neg neg neg pos neg neg neg neg pos
    ## [271] pos neg pos pos pos neg neg neg neg pos neg neg neg neg neg neg pos pos
    ## [289] pos neg neg neg neg neg neg neg neg pos pos pos neg pos neg neg neg neg
    ## [307] neg
    ## Levels: neg pos

``` r
## Classification----
## a.LDA classifier based on a 5-fold cross validation
train_control <- trainControl(method = "cv", number = 5)

PimaIndiansDiabetes_model_lda <-
  caret::train(`diabetes` ~ ., data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit, method = "lda2",
               metric = "Accuracy")
## Test the trained LDA model
predictions_lda <- predict(PimaIndiansDiabetes_model_lda,
                           PimaIndiansDiabetes_test[, 1:9])

## View the summary of the model and view the confusion matrix

print(PimaIndiansDiabetes_model_lda)
```

    ## Linear Discriminant Analysis 
    ## 
    ## 461 samples
    ##   8 predictor
    ##   2 classes: 'neg', 'pos' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 369, 369, 368, 369, 369 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa   
    ##   0.7571061  0.434539
    ## 
    ## Tuning parameter 'dimen' was held constant at a value of 1

``` r
caret::confusionMatrix(predictions_lda, PimaIndiansDiabetes_test$diabetes)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg 173  41
    ##        pos  27  66
    ##                                           
    ##                Accuracy : 0.7785          
    ##                  95% CI : (0.7278, 0.8237)
    ##     No Information Rate : 0.6515          
    ##     P-Value [Acc > NIR] : 8.978e-07       
    ##                                           
    ##                   Kappa : 0.4969          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.1149          
    ##                                           
    ##             Sensitivity : 0.8650          
    ##             Specificity : 0.6168          
    ##          Pos Pred Value : 0.8084          
    ##          Neg Pred Value : 0.7097          
    ##              Prevalence : 0.6515          
    ##          Detection Rate : 0.5635          
    ##    Detection Prevalence : 0.6971          
    ##       Balanced Accuracy : 0.7409          
    ##                                           
    ##        'Positive' Class : neg             
    ## 

``` r
PimaIndiansDiabetes_model_nb <-
  e1071::naiveBayes(diabetes ~ ., data = PimaIndiansDiabetes_train)


predictions_nb_e1071 <-
  predict(PimaIndiansDiabetes_model_nb, PimaIndiansDiabetes_test[, 1:9])

## View a summary of the naive Bayes model and the confusion matrix 
print(PimaIndiansDiabetes_model_nb)
```

    ## 
    ## Naive Bayes Classifier for Discrete Predictors
    ## 
    ## Call:
    ## naiveBayes.default(x = X, y = Y, laplace = laplace)
    ## 
    ## A-priori probabilities:
    ## Y
    ##       neg       pos 
    ## 0.6507592 0.3492408 
    ## 
    ## Conditional probabilities:
    ##      pregnant
    ## Y         [,1]     [,2]
    ##   neg 3.336667 3.031053
    ##   pos 4.677019 3.647949
    ## 
    ##      glucose
    ## Y         [,1]     [,2]
    ##   neg 111.4067 23.97660
    ##   pos 138.9255 32.78329
    ## 
    ##      pressure
    ## Y         [,1]     [,2]
    ##   neg 69.04333 17.44293
    ##   pos 70.84472 22.68330
    ## 
    ##      triceps
    ## Y         [,1]     [,2]
    ##   neg 19.77333 15.46607
    ##   pos 22.56522 17.04765
    ## 
    ##      insulin
    ## Y         [,1]      [,2]
    ##   neg  69.4400  93.16537
    ##   pos 105.8261 150.46634
    ## 
    ##      mass
    ## Y         [,1]     [,2]
    ##   neg 30.70833 7.842396
    ##   pos 35.24845 7.262568
    ## 
    ##      pedigree
    ## Y          [,1]      [,2]
    ##   neg 0.4322133 0.2842304
    ##   pos 0.5549565 0.3975045
    ## 
    ##      age
    ## Y         [,1]     [,2]
    ##   neg 31.21667 11.35780
    ##   pos 37.08696 11.25244

``` r
caret::confusionMatrix(predictions_nb_e1071, PimaIndiansDiabetes_test$diabetes)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg 169  38
    ##        pos  31  69
    ##                                           
    ##                Accuracy : 0.7752          
    ##                  95% CI : (0.7244, 0.8207)
    ##     No Information Rate : 0.6515          
    ##     P-Value [Acc > NIR] : 1.697e-06       
    ##                                           
    ##                   Kappa : 0.4974          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.4701          
    ##                                           
    ##             Sensitivity : 0.8450          
    ##             Specificity : 0.6449          
    ##          Pos Pred Value : 0.8164          
    ##          Neg Pred Value : 0.6900          
    ##              Prevalence : 0.6515          
    ##          Detection Rate : 0.5505          
    ##    Detection Prevalence : 0.6743          
    ##       Balanced Accuracy : 0.7449          
    ##                                           
    ##        'Positive' Class : neg             
    ## 

``` r
### Classification: SVM with Repeated k-fold Cross Validation----
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

PimaIndiansDiabetes_model_svm <-
  caret::train(diabetes ~ ., data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "svmLinearWeights2", metric = "Accuracy")


predictions_svm <- predict(PimaIndiansDiabetes_model_svm, PimaIndiansDiabetes_test[, 1:9])

#viewing of the prediction results 
print(PimaIndiansDiabetes_model_svm)
```

    ## L2 Regularized Linear Support Vector Machines with Class Weights 
    ## 
    ## 461 samples
    ##   8 predictor
    ##   2 classes: 'neg', 'pos' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 368, 369, 369, 369, 369, 369, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cost  Loss  weight  Accuracy   Kappa     
    ##   0.25  L1    1       0.5548075  0.07094989
    ##   0.25  L1    2       0.6028986  0.07928945
    ##   0.25  L1    3       0.6029531  0.03927952
    ##   0.25  L2    1       0.7064906  0.28315468
    ##   0.25  L2    2       0.7064828  0.39778026
    ##   0.25  L2    3       0.4648746  0.09767199
    ##   0.50  L1    1       0.4985741  0.05378535
    ##   0.50  L1    2       0.6051036  0.08829284
    ##   0.50  L1    3       0.6162381  0.10383272
    ##   0.50  L2    1       0.7115007  0.29837955
    ##   0.50  L2    2       0.7129032  0.40724901
    ##   0.50  L2    3       0.4641499  0.09680509
    ##   1.00  L1    1       0.5693548  0.07487652
    ##   1.00  L1    2       0.5395746  0.03548128
    ##   1.00  L1    3       0.5785180  0.05312346
    ##   1.00  L2    1       0.7165342  0.31167902
    ##   1.00  L2    2       0.6920212  0.37245454
    ##   1.00  L2    3       0.4641499  0.09680509
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were cost = 1, Loss = L2 and weight = 1.

``` r
caret::confusionMatrix(predictions_svm, PimaIndiansDiabetes_test$diabetes)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg 175  66
    ##        pos  25  41
    ##                                           
    ##                Accuracy : 0.7036          
    ##                  95% CI : (0.6491, 0.7541)
    ##     No Information Rate : 0.6515          
    ##     P-Value [Acc > NIR] : 0.03052         
    ##                                           
    ##                   Kappa : 0.2834          
    ##                                           
    ##  Mcnemar's Test P-Value : 2.751e-05       
    ##                                           
    ##             Sensitivity : 0.8750          
    ##             Specificity : 0.3832          
    ##          Pos Pred Value : 0.7261          
    ##          Neg Pred Value : 0.6212          
    ##              Prevalence : 0.6515          
    ##          Detection Rate : 0.5700          
    ##    Detection Prevalence : 0.7850          
    ##       Balanced Accuracy : 0.6291          
    ##                                           
    ##        'Positive' Class : neg             
    ## 

``` r
#training of the model
train_control <- trainControl(method = "LOOCV")

PimaIndiansDiabetes_model_nb_loocv <-
  caret::train(diabetes ~ ., data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "naive_bayes", metric = "Accuracy")

#testing of the model
predictions_nb_loocv <-
  predict(PimaIndiansDiabetes_model_nb_loocv, PimaIndiansDiabetes_test[, 1:9])

#viewing of the prediction results
print(PimaIndiansDiabetes_model_nb_loocv)
```

    ## Naive Bayes 
    ## 
    ## 461 samples
    ##   8 predictor
    ##   2 classes: 'neg', 'pos' 
    ## 
    ## No pre-processing
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 460, 460, 460, 460, 460, 460, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   usekernel  Accuracy   Kappa    
    ##   FALSE      0.7331887  0.3947225
    ##    TRUE      0.7418655  0.4109101
    ## 
    ## Tuning parameter 'laplace' was held constant at a value of 0
    ## Tuning
    ##  parameter 'adjust' was held constant at a value of 1
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were laplace = 0, usekernel = TRUE
    ##  and adjust = 1.

``` r
caret::confusionMatrix(predictions_nb_loocv, PimaIndiansDiabetes_test$diabetes)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg 176  39
    ##        pos  24  68
    ##                                           
    ##                Accuracy : 0.7948          
    ##                  95% CI : (0.7452, 0.8386)
    ##     No Information Rate : 0.6515          
    ##     P-Value [Acc > NIR] : 2.813e-08       
    ##                                           
    ##                   Kappa : 0.5329          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.07776         
    ##                                           
    ##             Sensitivity : 0.8800          
    ##             Specificity : 0.6355          
    ##          Pos Pred Value : 0.8186          
    ##          Neg Pred Value : 0.7391          
    ##              Prevalence : 0.6515          
    ##          Detection Rate : 0.5733          
    ##    Detection Prevalence : 0.7003          
    ##       Balanced Accuracy : 0.7578          
    ##                                           
    ##        'Positive' Class : neg             
    ## 
