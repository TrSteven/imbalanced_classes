library(mlbench)
library(PRROC)

MyPerformance <- function(model, test.set, which.plot = "none", print.conf = TRUE){
  ## This function returns different measures 
  ## for a model evaluated on the test set 
  
  ## model = the model that was trained
  
  ## test.set = the data set on which the model will be tested
  
  ## which.plot = if which.plot == "pr", pr-curve is plotted,
  ## and if which.plot == "roc", roc-curve is plotted
  
  pred.results <- predict(object = model, newdata = test.set)    
    
  # Calculate accuracy, precision, sensitivity, 
  # F-score, FPR and AUROC (Area Under ROC-curve)
  pred.performance <- mlr::performance(pred.results, measures = 
                                       list(acc, ppv, tpr, f1, fpr, mlr::auc))
  pred.performance <- as.list(pred.performance)
  pred.performance$fpr <- 1 -  pred.performance$fpr
  names(pred.performance) <- c("Accuracy", "Precision", 
                               "Sensitivity", "F1 score",
                               "Specificity", "AUROC")
  # Calculate AUPR (Area Under PR-curve)
  prob.pos <- pred.results$data$prob.pos
  observed.pos <- pred.results$data$truth == "pos"
  
  PRROC.pr <- PRROC::pr.curve(prob.pos[observed.pos], 
                              prob.pos[!observed.pos], curve=TRUE)
  
  PRROC.roc <- PRROC::roc.curve(prob.pos[observed.pos], 
                                prob.pos[!observed.pos], curve=TRUE)
  
  if(which.plot == "pr"){
    plot(PRROC.pr, col="black")   
  }else if(which.plot == "roc"){
    plot(PRROC.roc, col="black")  
  }
  
  if(print.conf){
    cat("\nConfusion matrix for:", model$task.desc$id, "\n")
    print(calculateConfusionMatrix(pred.results, sums = TRUE))
  }
  
  pred.performance$AUPR <- PRROC.pr$auc.davis.goadrich
    
  return(pred.performance)
}