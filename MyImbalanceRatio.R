MyImbalanceRatio <- function(dataset){
  # This function returns the Imbalance Ratio (IR) of a dataset
  # The column named "outcome" contains the variable that is predicted
  
  table.outcome <- table(dataset$outcome)
  IR <- max(table.outcome) / min(table.outcome)

  cat("Imbalance ratio is: ", IR, "\n", sep="")

  return(IR)
}
