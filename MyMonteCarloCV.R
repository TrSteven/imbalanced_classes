library(mlr)
library(UBL)
library(ROSE)
library(PRROC)
library(rpart)

source("MyTrainingTesting.R")

MyMonteCarloCV <- function(dataset, split, LearningAlg, 
                           numb.of.rounds, par.vals = list(), nbags){
  ## This function implements Monte Carlo cross-validation 
  ## with a certain split in each round, and returns the performace
  ## averaged over the different runs
  
  ## split = ratio that indicates how large the training set is compared to 
  ## the complete dataset
  
  ## numb.of.rounds = number of times the dataset is split into 
  ## a training and a test set
  
  ## par.vals = optional hyperparameters for learning algorithm
  
  df <- MyTrainingTesting(dataset = dataset, split = split, seed = 1,
                          LearningAlg = LearningAlg, par.vals = par.vals,
                          nbags = nbags)  
  
  if(numb.of.rounds != 1){
    for(i in 2:(numb.of.rounds)){
      df <- df + MyTrainingTesting(dataset = dataset, split = split, seed = i,
                                   LearningAlg = LearningAlg, 
                                   par.vals = par.vals, nbags = nbags)  
    }
  }  
  
  return(df/numb.of.rounds)
}
