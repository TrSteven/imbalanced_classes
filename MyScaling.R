MyScaling <- function(train.set, test.set, which.scaling){
  ## This function scales the numeric features of the training set 
  ## The values used to scale the training set are used to scale the test set
  
  ## Two possible scaling methods: 
  ## which.scaling = "min.max" --> scaling to range [0,1] 
  ## which.scaling = "standardization" --> substract with mean 
  ## and then divide by standard deviation 
  
  train.sc <- train.set
  test.sc <- test.set
  id.num <- unname(which(sapply(train.set, is.numeric)))
  
  if(which.scaling == "min.max"){
    
    for(i in id.num){
      train.max <- max(train.set[, i], na.rm = TRUE)
      train.min <- min(train.set[, i], na.rm = TRUE)
      diff <- train.max - train.min
      train.sc[, i] <- (train.set[, i] - train.min)/diff
      test.sc[, i] <- (test.set[, i] - train.min)/diff
    }  
    
  }else if(which.scaling == "standardization"){
    
    for(i in id.num){
      train.mean <- mean(train.set[, i], na.rm = TRUE)
      train.SD <- sd(train.set[, i], na.rm = TRUE)
      train.sc[, i] <- (train.set[, i] - train.mean)/train.SD
      test.sc[, i] <- (test.set[, i] - train.mean)/train.SD
    }  
    
  }else{
    
    stop("Wrong argument for which.scaling")
    
  }
  
  return(list(train.sc = train.sc,
              test.sc = test.sc))

}

