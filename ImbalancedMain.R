rm(list=ls(all = TRUE))

library(mlr)
library(mlbench)
library(UBL)
library(ROSE)
library(PRROC)
library(rpart)
library(C50)
library(xtable)

source("MyMonteCarloCV.R")
source("MyImbalanceRatio.R")

method <- 1
nbags <- 25
numb.of.rounds <- 10
split <- 1/2

if(method == 1){
  
  LearningAlg = "classif.rpart"
  par.vals = list(xval=10)
  part.caption = "decision trees"
  
} else if(method == 2){
  
  LearningAlg = "classif.boosting"
  par.vals = list(mfinal = 10)
  part.caption = "AdaBoost"
  
} else if(method == 3){
  
  LearningAlg = "classif.svm"
  par.vals = list()
  part.caption = "SVM"
  
}

file.conn <- file(paste0("tables/settings_method_", method, ".txt"))
writeLines(c(paste("Number of bags:", nbags), 
             paste("Number of rounds:", numb.of.rounds),
             paste("Split training/test:", split),
             paste("Learning algorithm:", LearningAlg),
             paste("Parameters:", "\n", names(par.vals), "\n", par.vals)), 
           file.conn)
close(file.conn)

##################
# Satellite data # 
##################   
data(Satellite)
data.sat <- Satellite
names(data.sat)[names(data.sat)=="classes"] <- "outcome"
table(data.sat$outcome) 
data.sat$outcome <- as.factor(ifelse(data.sat$outcome=="vegetation stubble", 
                                     "pos", "neg"))
table(data.sat$outcome)
MyImbalanceRatio(data.sat)

df.sat <- MyMonteCarloCV(dataset = data.sat, split = split, 
                         LearningAlg = LearningAlg, 
                         numb.of.rounds = numb.of.rounds,
                         par.vals = par.vals, nbags = nbags)

res.sat <- xtable(df.sat, align = rep("c",8), digits=3, 
           caption = paste0("Results for the satellite dataset using ", part.caption, "."))
print(res.sat, file = paste0("tables/", "satellite ", part.caption, ".tex"), 
      table.placement="H", size = "\\small")
round(df.sat,3)

###########
# Vehicle #
###########
data(Vehicle)
data.veh <- Vehicle
names(data.veh)[names(data.veh)=="Class"] <- "outcome"
table(data.veh$outcome)
data.veh$outcome <- as.factor(ifelse(data.veh$outcome=="van", 
                                     "pos", "neg"))
table(data.veh$outcome)
MyImbalanceRatio(data.veh)

df.veh <- MyMonteCarloCV(dataset = data.veh, split = split, 
                         LearningAlg = LearningAlg, 
                         numb.of.rounds = numb.of.rounds,
                         par.vals = par.vals)

res.veh <- xtable(df.veh, align = rep("c",8), digits=3, 
           caption = paste0("Results for the vehicle dataset using ", part.caption, "."))
print(res.veh, file = paste0("tables/", "vehicle ", part.caption, ".tex"),
      table.placement="H", size = "\\small")
round(df.veh,3)

###########
# abalone #
###########
link.abalone <- "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
data.aba <- read.csv(link.abalone, header = FALSE)
names(data.aba) <- c("sex", "length", "diameter", "height", "weight.whole",
                     "weight.shucked", "weight.viscera", "weight.shell", "rings")
names(data.aba)[names(data.aba)=="rings"] <- "outcome"
cut.ada <- quantile(data.aba$outcome, probs = 0.90)
data.aba$outcome <- as.factor(ifelse(data.aba$outcome >= cut.ada, 
                                     "pos", "neg"))
table(data.aba$outcome)
MyImbalanceRatio(data.aba)

df.aba <- MyMonteCarloCV(dataset = data.aba, split = split, 
                         LearningAlg = LearningAlg, 
                         numb.of.rounds = numb.of.rounds,
                         par.vals = par.vals)

res.aba <- xtable(df.aba, align = rep("c",8), digits=3, 
           caption = paste0("Results for the abalone dataset using ", part.caption, "."))
print(res.aba, file = paste0("tables/", "abalone ", part.caption, ".tex"),
      table.placement="H", size = "\\small")
round(df.aba, 3)

#############
# wine data #
#############
data.wine <- read.csv("winequality-red.csv", header = TRUE, sep = ";")

names(data.wine)[names(data.wine)=="quality"] <- "outcome"
data.wine$outcome <- as.factor(ifelse(data.wine$outcome >= 7, 
                                      "pos", "neg"))
table(data.wine$outcome)
MyImbalanceRatio(data.wine)

df.wine <- MyMonteCarloCV(dataset = data.wine, split = split, 
                          LearningAlg = LearningAlg, 
                          numb.of.rounds = numb.of.rounds,
                          par.vals = par.vals)

res.wine <- xtable(df.wine, align = rep("c",8), digits=3, 
            caption = paste0("Results for the wine dataset using ", part.caption, "."))
print(res.wine, file = paste0("tables/", "wine ", part.caption, ".tex"),
      table.placement="H", size = "\\small")
round(df.wine, 3)

##########
# letter #
##########
data(LetterRecognition)
set.seed(1)
data.let <- LetterRecognition[sample(nrow(LetterRecognition), 2000, replace = FALSE), ]
names(data.let)[names(data.let)=="lettr"] <- "outcome"
table(data.let$outcome)
data.let$outcome <- as.factor(ifelse(data.let$outcome == "I", 
                                     "pos", "neg"))
table(data.let$outcome)
MyImbalanceRatio(data.let)

df.let <- MyMonteCarloCV(dataset = data.let, split = split, 
                         LearningAlg = LearningAlg, 
                         numb.of.rounds = numb.of.rounds,
                         par.vals = par.vals)

res.let <- xtable(df.let, align = rep("c",8), digits=3, 
           caption = paste0("Results for the letter dataset using ", part.caption, "."))
print(res.let, file = paste0("tables/", "letter ", part.caption, ".tex"),
      table.placement="H", size = "\\small")
round(df.let, 3)

###########################################################

if(method == 1){
  
  save(LearningAlg, df.sat, df.veh, df.aba, df.wine, df.let, file="rpart.RData")
  
} else if(method == 2){
  
  save(LearningAlg, df.sat, df.veh, df.aba, df.wine, df.let, file="boosting.RData")
  
} else if(method == 3){
  
  save(LearningAlg, df.sat, df.veh, df.aba, df.wine, df.let, file="svm.RData")
  
}