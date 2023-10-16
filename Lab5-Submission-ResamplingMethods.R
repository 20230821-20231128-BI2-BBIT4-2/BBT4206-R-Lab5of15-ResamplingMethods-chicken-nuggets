if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

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
caret::confusionMatrix(predictions_nb_e1071,
                       PimaIndiansDiabetes_test$diabetes)


##confusion matrix
plot(table(predictions_nb_e1071,
           PimaIndiansDiabetes_test$diabetes))


##Bootstrapping----


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
print(predictions_lm)


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



##(CV, Repeated CV, and LOOCV)----
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
print(predictions_lm)



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
caret::confusionMatrix(predictions_lda, PimaIndiansDiabetes_test$diabetes)


PimaIndiansDiabetes_model_nb <-
  e1071::naiveBayes(diabetes ~ ., data = PimaIndiansDiabetes_train)


predictions_nb_e1071 <-
  predict(PimaIndiansDiabetes_model_nb, PimaIndiansDiabetes_test[, 1:9])

## View a summary of the naive Bayes model and the confusion matrix 
print(PimaIndiansDiabetes_model_nb)
caret::confusionMatrix(predictions_nb_e1071, PimaIndiansDiabetes_test$diabetes)

### Classification: SVM with Repeated k-fold Cross Validation----
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

PimaIndiansDiabetes_model_svm <-
  caret::train(diabetes ~ ., data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "svmLinearWeights2", metric = "Accuracy")


predictions_svm <- predict(PimaIndiansDiabetes_model_svm, PimaIndiansDiabetes_test[, 1:9])


print(PimaIndiansDiabetes_model_svm)
caret::confusionMatrix(predictions_svm, PimaIndiansDiabetes_test$diabetes)

##Classification: Naive Bayes with Leave One Out Cross Validation ----
train_control <- trainControl(method = "LOOCV")

PimaIndiansDiabetes_model_nb_loocv <-
  caret::train(diabetes ~ ., data = PimaIndiansDiabetes_train,
               trControl = train_control, na.action = na.omit,
               method = "naive_bayes", metric = "Accuracy")


predictions_nb_loocv <-
  predict(PimaIndiansDiabetes_model_nb_loocv, PimaIndiansDiabetes_test[, 1:9])


print(PimaIndiansDiabetes_model_nb_loocv)
caret::confusionMatrix(predictions_nb_loocv, PimaIndiansDiabetes_test$diabetes)
