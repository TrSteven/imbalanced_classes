library(mlr)

MyModel <- function(LearningAlg, train.set, nameTask, par.vals = list(),
                    bagging.method = "none", nbags = nbags){
  ## LearningAlg = the algorithm used to predict the outcome,
  ## e.g. "classif.logreg", "classif.naiveBayes", 
  ## "classif.rpart", "classif.svm"
    
  ## train.set = dataframe containing the training set,
  ## where the column named "outcome" contains the variable that is predicted,
  ## and this variable is a factor with levels "pos" 
  ## (i.e. the positive instances) and "neg" (i.e. the negative instances)
    
  ## par.vals = optional list of named hyperparameters
    
  ## bagging.method = an (optional) bagging method,
  ## possible values: "none", "bagging", "overbagging"
    
  ## nbags = number of bagging rounds
    
  # construct the classification task
  task <- makeClassifTask(data = train.set, target = "outcome", 
                          id = nameTask, positive = "pos")
    
  # construct the learner
  if(bagging.method == "none"){
        
    lrn <- makeLearner(cl = LearningAlg, predict.type = "prob", 
                       par.vals = par.vals) 
        
  }else if(bagging.method == "bagging"){
        
    lrn <- makeLearner(cl = LearningAlg, predict.type = "response",
                       par.vals = par.vals)
        
    lrn <- makeBaggingWrapper(lrn, bw.iters = nbags, bw.feats = 1,
                              bw.replace = TRUE, bw.size = 1)
        
    lrn <- setPredictType(lrn, "prob")

        
  }else if(bagging.method == "overbagging"){
        
    lrn <- makeLearner(cl = LearningAlg, predict.type = "response",
                       par.vals = par.vals)
        
    # ratio of number of instances in majority class vs. minority class
    max.min.ratio <- max(table(train.set$outcome)) /
                     min(table(train.set$outcome))
    
    lrn <- makeOverBaggingWrapper(lrn, obw.rate = max.min.ratio, 
                                  obw.iters = nbags)
        
    lrn <- setPredictType(lrn, "prob")
        
  }else{
    stop("Wrong argument for ensemble.method.")
  }
    
  # train the model
  cat("Bagging method: ", bagging.method, "\n", sep = "")
  cat("Hyperparameters: \n", sep = "")
  print(unlist(getHyperPars(lrn)))
  cat("Training the model: ", LearningAlg, "\n", sep = "")
  ptm <- proc.time()
  model <- train(lrn, task)
  ptm.end <- proc.time() - ptm
  cat("Time elapsed: ", as.numeric(ptm.end["elapsed"]), "s\n", sep="")
  cat("\n")
    
  return(model)
}