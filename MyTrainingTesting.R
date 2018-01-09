library(mlr)
library(UBL)
library(ROSE)
library(PRROC)
library(rpart)

MyTrainingTesting <- function(dataset, split, LearningAlg, seed, 
                              par.vals = list(), nbags = nbags){
  ## This functions returns a dataframe with 
  ## the performance measures of different sampling methods
  
  ## dataset = the dataset that will be split into a training and test set,
  ## where the column named "outcome" contains the variable that is predicted,
  ## and this variable is a factor with levels "pos" 
  ## (i.e. the positive instances) and "neg" (i.e. the negative instances)
  
  ## LearningAlg = the learning algorithm (e.g. "classif.logreg", 
  ## "classif.naiveBayes", "classif.rpart", "classif.svm")
  
  ## split = the ratio used to split the data into a training and test set
  
  ## seed = the seed that will be used
  
  ## par.vals = optional list of named hyperparameters
  
  # Set the seed
  set.seed(seed)
  
  # Create training and test set
  # using stratified Monte Carlo cross-validation
  n <- nrow(dataset)
  pos.ID <- which(dataset$outcome == "pos")
  neg.ID <- setdiff(1:n, pos.ID)
  
  train.pos.ID <- sample(pos.ID, size = split*length(pos.ID), replace=FALSE)
  train.neg.ID <- sample(neg.ID, size = split*length(neg.ID), replace=FALSE)
  train.ID <- c(train.pos.ID, train.neg.ID)
  
  train.set <- dataset[train.ID, ]
  test.set <- dataset[-train.ID, ] 
  
  cat("Training set: \n")
  print(table(train.set$outcome))
  cat("Test set: \n")
  print(table(test.set$outcome))
  
  # Scale the training and test set
  source("MyScaling.R")
  scaled.data <- MyScaling(train.set = train.set, test.set = test.set, 
                           which.scaling = "min.max")
  train.set <- scaled.data$train.sc
  test.set <- scaled.data$test.sc
  
  # Create more balanced training datasets
  # Use HEOM distance (because predictors can be categorical or continuous)
  # If only continuous predictors, one could also use "Euclidean"
  dist <- "HEOM"
  
  cat("Creating more balanced training sets \n")
  cat("Number of instances in original training set:\n")
  print(table(train.set$outcome))
  
  cat("\nExecuting ROS\n")
  train.ROS <- RandOverClassif(outcome ~ ., train.set)
  print(table(train.ROS$outcome))
  
  cat("\nExecuting RUS \n")
  train.RUS <- RandUnderClassif(outcome ~ ., train.set)
  print(table(train.RUS$outcome))
  
  cat("\nExecuting SMOTE \n")
  train.SMOTE <- SmoteClassif(outcome ~ ., train.set, dist = dist)
  print(table(train.SMOTE$outcome))
  
  cat("\nExecuting ENN \n")
  train.ENN <- ENNClassif(outcome ~ ., train.set, dist = dist)[[1]]
  print(table(train.ENN$outcome))
  
  cat("\nExecuting Tomek \n")
  train.Tomek <- TomekClassif(outcome ~ ., train.set, dist = dist)[[1]]
  print(table(train.Tomek$outcome))

  cat("\nExecuting Gauss \n")
  train.Gauss <- GaussNoiseClassif(outcome ~ .,train.set)
  print(table(train.Gauss$outcome))
  cat("\n")
  
  cat("\nExecuting ROSE \n")
  train.ROSE <- ROSE(outcome ~ ., data=train.set, p=0.5)$data
  print(table(train.ROSE$outcome))
  cat("\n")
  
  # Train the model
  source("MyModel.R")
  
  list.models <- list(
    model.none = MyModel(LearningAlg = LearningAlg, train.set = train.set, 
                         nameTask = "None", bagging.method="none", 
                         par.vals = par.vals),
    model.ROS = MyModel(LearningAlg = LearningAlg, train.set = train.ROS, 
                        nameTask = "ROS", bagging.method="none", 
                        par.vals = par.vals),
    model.RUS = MyModel(LearningAlg = LearningAlg, train.set = train.RUS, 
                        nameTask = "RUS", bagging.method="none", 
                        par.vals = par.vals),
    model.SMOTE = MyModel(LearningAlg = LearningAlg, train.set = train.SMOTE, 
                          nameTask = "SMOTE", bagging.method="none", 
                          par.vals = par.vals),
    model.ENN = MyModel(LearningAlg = LearningAlg, train.set = train.ENN, 
                        nameTask = "ENN", bagging.method="none", par.vals = 
                          par.vals),
    model.Tomek = MyModel(LearningAlg = LearningAlg, train.set = train.Tomek, 
                          nameTask = "Tomek", bagging.method="none", 
                          par.vals = par.vals),
    model.Gauss = MyModel(LearningAlg = LearningAlg, train.set = train.Gauss, 
                          nameTask = "Gauss", bagging.method="none", 
                          par.vals = par.vals),
    model.ROSE = MyModel(LearningAlg = LearningAlg, train.set = train.ROSE, 
                         nameTask = "ROSE", bagging.method="none", 
                         par.vals = par.vals),
    model.bag = MyModel(LearningAlg = LearningAlg, train.set = train.set, 
                        nameTask = "Bagging", bagging.method = "bagging", 
                        nbags = nbags, par.vals = par.vals),
    model.overbag = MyModel(LearningAlg = LearningAlg, train.set = train.set, 
                            nameTask = "Overbagging", 
                            bagging.method = "overbagging", nbags = nbags, 
                            par.vals = par.vals)
  )
  
  source("MyPerformance.R")
  
  df.performance <- NULL
  for(i in 1:length(list.models)){
    temp <- as.data.frame(MyPerformance(model = list.models[[i]], 
                                        test.set = test.set))
    rownames(temp) <- list.models[[i]]$task.desc$id
    df.performance <- rbind(df.performance, temp)
  }
  return(df.performance)
}