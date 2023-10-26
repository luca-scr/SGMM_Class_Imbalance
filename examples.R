# install.packages(c("mclust", "caret", "themis", "ModelMetrics", "data.table", "ggplot2"))

library(mclust)
library(MASS)
library(caret)
library(themis)
library(ModelMetrics)
library(data.table)
library(ggplot2)
theme_set(theme_bw())

balancedAccuracy <- function(actual, predicted, cutoff = 0.5, ...) 
{
  0.5*(ModelMetrics::sensitivity(actual, predicted, cutoff) +
	     ModelMetrics::specificity(actual, predicted, cutoff) )
}

YoudenScore <- function(actual, predicted, cutoff = 0.5, ...) 
{
  ModelMetrics::sensitivity(actual, predicted, cutoff) + 
  ModelMetrics::specificity(actual, predicted, cutoff) -1 
}

evalMetrics <- function(obs, prob, threshold = 0.5)
{  
  c(ModelMetrics::ce(obs, ifelse(prob > threshold, 1, 0)),
    ModelMetrics::kappa(obs, prob, threshold),
    ModelMetrics::logLoss(obs, prob),
    ModelMetrics::brier(obs, prob),
    ModelMetrics::precision(obs, prob, threshold),
    ModelMetrics::recall(obs, prob, threshold),
    ModelMetrics::fScore(obs, prob, threshold, beta = 1),
    ModelMetrics::fScore(obs, prob, threshold, beta = 2),
    ModelMetrics::sensitivity(obs, prob, threshold),
    ModelMetrics::specificity(obs, prob, threshold),
    balancedAccuracy(obs, prob, threshold),
    YoudenScore(obs, prob, threshold),
    ModelMetrics::auc(obs, prob))
}


# Simulated example 1 --------------------------------------------------

data_sim <- function(n, pro)
{
  # n = sample size
  # pro = prob minority class
  n <- as.integer(n)
  pro <- max(0, min(pro, 1))
  y = sample(0:1, size = n, replace = TRUE, prob = c(1-pro, pro))
  x = matrix(as.double(NA), nrow = n, ncol = 2)
  x[y==0,] = mvrnorm(sum(y==0), 
                     mu = c(0,0), 
                     Sigma = diag(2))
  x[y==1,] = mvrnorm(sum(y==1), 
                     mu = c(1,1), 
                     Sigma = matrix(c(1,-0.5,-0.5,1),2,2))
  list(x = x, y = y, class = as.factor(y))
}

nsim = 100
out = array(as.double(NA), dim = c(nsim, 14, 13))
dimnames(out) = list(NULL, 
                     c("EDDA", "EDDA+costSens",
                       "EDDA+adjPriorThreshold",
                       "EDDDA+optThreshold",
                       "EDDA+downSamp", "EDDA+upSamp", 
                       "EDDA+smote",
                       "MCLUSTDA", "MCLUSTDA+costSens",
                       "MCLUSTDA+adjPriorThreshold",
                       "MCLUSTDA+optThreshold",
                       "MCLUSTDA+downSamp", "MCLUSTDA+upSamp", 
                       "MCLUSTDA+smote"),
                     c("CE", "kappa", "logLoss", "Brier",
                       "Precision", "Recall", "F1", "F2",
                       "Sensitivity/TPR", "Specificity/TNR", 
                       "BalancedAccuracy", "YoudenScore", "AUC"))

pro = 0.1
# pro = 0.01
n = 5000
test_mult = 10

for(j in 1:nsim)
{
  cat(".")
  # data
  train = data_sim(n = n, pro = pro)
  test = data_sim(n = test_mult*n, pro = pro)
  train_downSamp = downSample(x = train$x, y = train$class, list = TRUE); names(train_downSamp)[2] = "class"
  train_upSamp = upSample(x = train$x, y = train$class, list = TRUE); names(train_upSamp)[2] = "class"
  train_smote = smote(as.data.frame(train), "class")
  train_smote = list(x = train_smote[,1:2], class = train_smote$class)
  
  # Imbalance Ratio
  p = prop.table(table(train$class))
  tau1 = min(p)
  IR = max(p)/min(p)

  # fit EDDA model on unbalanced data
  EDDA = MclustDA(train$x, train$class, modelType = "EDDA")
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,1,] = evalMetrics(test$y, prob_pred)
  
  # fit EDDA model on unbalanced data + cost sensitive predictions
  pred = predict(EDDA, newdata = test$x, prop = EDDA$prop * c(1,IR))
  prob_pred = pred$z[,"1"]
  out[j,2,] = evalMetrics(test$y, prob_pred)

  # fit EDDA model on unbalanced data + threshold by adj prior prob
  adjPriorProbs = classPriorProbs(EDDA, train$x)
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,3,] = evalMetrics(test$y, prob_pred, adjPriorProbs["1"])

  # fit EDDA model on unbalanced data + optimal threshold by CV
  threshold = seq(0, 1, by = 0.01)
  metrics = matrix(as.double(NA), nrow = 13, ncol = length(threshold))
  cv = cvMclustDA(EDDA)
  prob_pred = cv$z[,"1"]
  for(i in 1:length(threshold))
  { metrics[,i] = evalMetrics(train$y, prob_pred, threshold[i]) }
  rownames(metrics) = dimnames(out)[[3]]
  # plot(threshold, metrics["BalancedAccuracy",])
  optThreshold = threshold[which.max(metrics["BalancedAccuracy",])]
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,4,] = evalMetrics(test$y, prob_pred, optThreshold)
  
  # fit EDDA model on downsampled data
  EDDA = MclustDA(train_downSamp$x, train_downSamp$class, modelType = "EDDA")
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,5,] = evalMetrics(test$y, prob_pred)
  
  # fit EDDA model on upsampled data
  EDDA = MclustDA(train_upSamp$x, train_upSamp$class, modelType = "EDDA")
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,6,] = evalMetrics(test$y, prob_pred)
  
  # fit EDDA model on smote data
  EDDA = MclustDA(train_smote$x, train_smote$class, modelType = "EDDA")
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,7,] = evalMetrics(test$y, prob_pred)
  
  # fit MCLUSTDA model on unbalanced data
  MCLUSTDA = MclustDA(train$x, train$class, modelType = "MclustDA")
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,8,] = evalMetrics(test$y, prob_pred)
  
  # fit MCLUSTDA model on unbalanced data + cost senstive predictions
  pred = predict(MCLUSTDA, newdata = test$x, prop = MCLUSTDA$prop * c(1,IR))
  prob_pred = pred$z[,"1"]
  out[j,9,] = evalMetrics(test$y, prob_pred)
  
  # fit MCLUSTDA model on unbalanced data + threshold by adj prior prob
  adjPriorProbs = classPriorProbs(MCLUSTDA, train$x)
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,10,] = evalMetrics(test$y, prob_pred, adjPriorProbs["1"])

  # fit MCLUSTDA model on unbalanced data + optimal threshold by CV
  threshold = seq(0, 1, by = 0.01)
  metrics = matrix(as.double(NA), nrow = 13, ncol = length(threshold))
  cv = cvMclustDA(MCLUSTDA)
  prob_pred = cv$z[,"1"]
  for(i in 1:length(threshold))
  { metrics[,i] = evalMetrics(train$y, prob_pred, threshold[i]) }
  rownames(metrics) = dimnames(out)[[3]]
  # plot(threshold, metrics["BalancedAccuracy",])
  optThreshold = threshold[which.max(metrics["BalancedAccuracy",])]
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,11,] = evalMetrics(test$y, prob_pred, optThreshold)

  # fit MCLUSTDA model on downsampled data
  MCLUSTDA = MclustDA(train_downSamp$x, train_downSamp$class, modelType = "MclustDA")
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,12,] = evalMetrics(test$y, prob_pred)
  
  # fit MCLUSTDA model on upsampled data
  MCLUSTDA = MclustDA(train_upSamp$x, train_upSamp$class, modelType = "MclustDA")
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,13,] = evalMetrics(test$y, prob_pred)
  
  # fit MCLUSTDA model on smote data
  MCLUSTDA = MclustDA(train_smote$x, train_smote$class, modelType = "MclustDA")
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,14,] = evalMetrics(test$y, prob_pred)
}

apply(out[,,1:4], 3, colMeans)
apply(out[,,5:8], 3, colMeans)
apply(out[,,9:12], 3, colMeans)

# Simulated example 2 --------------------------------------------------

data_sim <- function(n, pro)
{
  # n = sample size
  # pro = prob minority class
  n <- as.integer(n)
  pro <- max(0, min(pro, 1))
  y = sample(0:1, size = n, replace = TRUE, prob = c(1-pro, pro))
  x = matrix(as.double(NA), nrow = n, ncol = 10)
  x[y==0,] = mvrnorm(sum(y==0), 
                     mu = rep(0, 10), 
                     Sigma = c(0.25,rep(1,9))*diag(10))
  ok = FALSE
  n1 = sum(y==1)
  xx = matrix(as.double(NA), nrow = 0, ncol = 10)
  while(!ok)
  {
    xx = rbind(xx, mvrnorm(n1, mu = rep(0,10), Sigma = diag(10)))
    cond = which(sqrt(apply(xx^2, 1, sum)) > 4 & xx[,1] <= 0 )
    xx = xx[cond,,drop=FALSE]
    if(nrow(xx) > n1) ok = TRUE
  }
  x[y==1,] = xx[1:n1,]
    
  list(x = x, y = y, class = as.factor(y))
}

nsim = 100
out = array(as.double(NA), dim = c(nsim, 14, 13))
dimnames(out) = list(NULL, 
                     c("EDDA", "EDDA+costSens",
                       "EDDA+adjPriorThreshold",
                       "EDDDA+optThreshold",
                       "EDDA+downSamp", "EDDA+upSamp", 
                       "EDDA+smote", 
                       "MCLUSTDA", "MCLUSTDA+costSens",
                       "MCLUSTDA+adjPriorThreshold",
                       "MCLUSTDA+optThreshold",
                       "MCLUSTDA+downSamp", "MCLUSTDA+upSamp", 
                       "MCLUSTDA+smote"),
                     c("CE", "kappa", "logLoss", "Brier",
                       "Precision", "Recall", "F1", "F2",
                       "Sensitivity/TPR", "Specificity/TNR", 
                       "BalancedAccuracy", "YoudenScore", "AUC"))

pro = 0.1
# pro = 0.01
n = 5000
test_mult = 10

for(j in 1:nsim)
{
  cat(".")
  # data
  train = data_sim(n = n, pro = pro)
  train_downSamp = downSample(x = train$x, y = train$class, list = TRUE); names(train_downSamp)[2] = "class"
  train_upSamp = upSample(x = train$x, y = train$class, list = TRUE); names(train_upSamp)[2] = "class"
  train_smote = smote(as.data.frame(train), "class")
  train_smote = list(x = train_smote[,1:10], class = train_smote$class)
  test = data_sim(n = test_mult*n, pro = pro)
  
  # Imbalance Ratio
  p = prop.table(table(train$class))
  tau1 = min(p)
  IR = min(p)/max(p)
      
  # fit EDDA model on unbalanced data
  EDDA = MclustDA(train$x, train$class, modelType = "EDDA")
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,1,] = evalMetrics(test$y, prob_pred)
  
  # fit EDDA model on unbalanced data + cost senstive predictions
  pred = predict(EDDA, newdata = test$x, prop = EDDA$prop * c(1,1/IR))
  prob_pred = pred$z[,"1"]
  out[j,2,] = evalMetrics(test$y, prob_pred)

  # fit EDDA model on unbalanced data + threshold by adj prior prob
  adjPriorProbs = classPriorProbs(EDDA, train$x)
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,3,] = evalMetrics(test$y, prob_pred, adjPriorProbs["1"])

  # fit EDDA model on unbalanced data + optimal threshold by CV
  threshold = seq(0, 1, by = 0.01)
  metrics = matrix(as.double(NA), nrow = 13, ncol = length(threshold))
  cv = cvMclustDA(EDDA)
  prob_pred = cv$z[,"1"]
  for(i in 1:length(threshold))
  { metrics[,i] = evalMetrics(train$y, prob_pred, threshold[i]) }
  rownames(metrics) = dimnames(out)[[3]]
  # plot(threshold, metrics["BalancedAccuracy",])
  optThreshold = threshold[which.max(metrics["BalancedAccuracy",])]
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,4,] = evalMetrics(test$y, prob_pred, optThreshold)
	
  # fit EDDA model on downsampled data
  EDDA = MclustDA(train_downSamp$x, train_downSamp$class, modelType = "EDDA")
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,5,] = evalMetrics(test$y, prob_pred)
  
  # fit EDDA model on upsampled data
  EDDA = MclustDA(train_upSamp$x, train_upSamp$class, modelType = "EDDA")
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,6,] = evalMetrics(test$y, prob_pred)
  
  # fit EDDA model on smote data
  EDDA = MclustDA(train_smote$x, train_smote$class, modelType = "EDDA")
  pred = predict(EDDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,7,] = evalMetrics(test$y, prob_pred)
  
  # fit MCLUSTDA model
  MCLUSTDA = MclustDA(train$x, train$class, modelType = "MclustDA")
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,8,] = evalMetrics(test$y, prob_pred)
  
  # fit MCLUSTDA model on unbalanced data + cost senstive predictions
  pred = predict(MCLUSTDA, newdata = test$x, prop = MCLUSTDA$prop * c(1,1/IR))
  prob_pred = pred$z[,"1"]
  out[j,9,] = evalMetrics(test$y, prob_pred)
	
  # fit MCLUSTDA model on unbalanced data + threshold by adj prior prob
  adjPriorProbs = classPriorProbs(MCLUSTDA, train$x)
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,10,] = evalMetrics(test$y, prob_pred, adjPriorProbs["1"])

  # fit MCLUSTDA model on unbalanced data + optimal threshold by CV
  threshold = seq(0, 1, by = 0.01)
  metrics = matrix(as.double(NA), nrow = 13, ncol = length(threshold))
  cv = cvMclustDA(MCLUSTDA)
  prob_pred = cv$z[,"1"]
  for(i in 1:length(threshold))
  { metrics[,i] = evalMetrics(train$y, prob_pred, threshold[i]) }
  rownames(metrics) = dimnames(out)[[3]]
  # plot(threshold, metrics["BalancedAccuracy",])
  optThreshold = threshold[which.max(metrics["BalancedAccuracy",])]
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,11,] = evalMetrics(test$y, prob_pred, optThreshold)

  # fit MCLUSTDA model on downsampled data
  MCLUSTDA = MclustDA(train_downSamp$x, train_downSamp$class, modelType = "MclustDA")
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,12,] = evalMetrics(test$y, prob_pred)
  
  # fit MCLUSTDA model on upsampled data
  MCLUSTDA = MclustDA(train_upSamp$x, train_upSamp$class, modelType = "MclustDA")
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,13,] = evalMetrics(test$y, prob_pred)
  
  # fit MCLUSTDA model on smote data
  MCLUSTDA = MclustDA(train_smote$x, train_smote$class, modelType = "MclustDA")
  pred = predict(MCLUSTDA, newdata = test$x)
  prob_pred = pred$z[,"1"]
  out[j,14,] = evalMetrics(test$y, prob_pred)
}

apply(out[,,1:4], 3, colMeans)
apply(out[,,5:8], 3, colMeans)
apply(out[,,9:12], 3, colMeans)

# Wine quality data example --------------------------------------------

data = fread("wine_quality_white.csv")
data[, quality := factor(ifelse(quality < 8, 0, 1), 
                         labels = c("MedLow", "High"))]
tab = table(data$quality)
cbind(Count = tab, "%" = prop.table(tab)*100)

set.seed(20230922)

train = caret::createDataPartition(1:nrow(data), p = 2/3)[[1]]
# training dataset
data_train = data[train,]
# test dataset
data_test = data[-train,]

# Dummy variable for classes
data_train[, y := ifelse(quality == "High", 1, 0)]
data_test[, y := ifelse(quality == "High", 1, 0)]

# Imbalance Ratio
p = prop.table(table(data_train$quality))
IR = p["MedLow"]/p["High"]
IR

# Sampling
data_downSamp = caret::downSample(x = data_train[,1:11], 
                                  y = data_train$quality)
names(data_downSamp)[ncol(data_downSamp)] = "quality"
data_upSamp = caret::upSample(x = data_train[,1:11], 
                              y = data_train$quality)
names(data_upSamp)[ncol(data_upSamp)] = "quality"
data_smote = themis::smote(df = data_train[,1:12], 
                           var = "quality")
data_smote[, y := ifelse(quality == "High", 1, 0)]

out = array(as.double(NA), dim = c(14, 13))
dimnames(out) = list(c("EDDA", "EDDA+costSens",
                       "EDDA+adjPriorThreshold",
                       "EDDDA+optThreshold",
                       "EDDA+downSamp", "EDDA+upSamp", 
                       "EDDA+smote", 
                       "MCLUSTDA", "MCLUSTDA+costSens",
                       "MCLUSTDA+adjPriorThreshold",
                       "MCLUSTDA+optThreshold",
                       "MCLUSTDA+downSamp", "MCLUSTDA+upSamp", 
                       "MCLUSTDA+smote"),
                     c("CE", "kappa", "logLoss", "Brier",
                       "Precision", "Recall", "F1", "F2",
                       "Sensitivity/TPR", "Specificity/TNR", 
                       "BalancedAccuracy", "YoudenScore", "AUC"))

# fit EDDA model on unbalanced data
EDDA1 = MclustDA(data_train[,1:11], data_train$quality, modelType = "EDDA")
pred = predict(EDDA1, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[1,] = evalMetrics(data_test$y, prob_pred)

# fit EDDA model on unbalanced data + cost senstive predictions
pred = predict(EDDA1, newdata = data_test[,1:11], 
               prop = EDDA1$prop * c(1,IR))
prob_pred = pred$z[,"High"]
out[2,] = evalMetrics(data_test$y, prob_pred)

# fit EDDA model on unbalanced data + threshold by adj prior prob
adjPriorProbs_EDDA = classPriorProbs(EDDA1, data_train[,1:11])
pred = predict(EDDA1, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[3,] = evalMetrics(data_test$y, prob_pred, adjPriorProbs_EDDA["High"])

# fit EDDA model on unbalanced data + optimal threshold by CV
threshold = seq(0, 1, by = 0.01)
metrics = matrix(as.double(NA), nrow = 13, ncol = length(threshold))
cv = cvMclustDA(EDDA1)
prob_pred = cv$z[,"High"]
for(i in 1:length(threshold))
   { metrics[,i] = evalMetrics(data_train$y, prob_pred, threshold[i]) }
rownames(metrics) = dimnames(out)[[2]]
# plot(threshold, metrics["BalancedAccuracy",])
optThreshold_EDDA = threshold[which.max(metrics["BalancedAccuracy",])]
pred = predict(EDDA1, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[4,] = evalMetrics(data_test$y, prob_pred, optThreshold_EDDA)

# fit EDDA model on downsampled data
EDDA2 = MclustDA(data_downSamp[,1:11], data_downSamp$quality, modelType = "EDDA")
pred = predict(EDDA2, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[5,] = evalMetrics(data_test$y, prob_pred)

# fit EDDA model on upsampled data
EDDA3 = MclustDA(data_upSamp[,1:11], data_upSamp$quality, modelType = "EDDA")
pred = predict(EDDA3, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[6,] = evalMetrics(data_test$y, prob_pred)

# fit EDDA model on smote data
EDDA4 = MclustDA(data_smote[,1:11], data_smote$quality, modelType = "EDDA")
pred = predict(EDDA4, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[7,] = evalMetrics(data_test$y, prob_pred)

# fit MCLUSTDA model
MCLUSTDA1 = MclustDA(data_train[,1:11], data_train$quality, modelType = "MclustDA", G = 1:5)
pred = predict(MCLUSTDA1, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[8,] = evalMetrics(data_test$y, prob_pred)

# fit MCLUSTDA model on unbalanced data + cost senstive predictions
pred = predict(MCLUSTDA1, newdata = data_test[,1:11], 
               prop = MCLUSTDA1$prop * c(1,IR))
prob_pred = pred$z[,"High"]
out[9,] = evalMetrics(data_test$y, prob_pred)

# fit MCLUSTDA model on unbalanced data + threshold by adj prior prob
adjPriorProbs_MCLUSTDA = classPriorProbs(MCLUSTDA1, data_train[,1:11])
pred = predict(MCLUSTDA1, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[10,] = evalMetrics(data_test$y, prob_pred, adjPriorProbs_MCLUSTDA["High"])

# fit MCLUSTDA model on unbalanced data + optimal threshold by CV
threshold = seq(0, 1, by = 0.01)
metrics = matrix(as.double(NA), nrow = 13, ncol = length(threshold))
cv = cvMclustDA(MCLUSTDA1)
prob_pred = cv$z[,"High"]
for(i in 1:length(threshold))
   { metrics[,i] = evalMetrics(data_train$y, prob_pred, threshold[i]) }
rownames(metrics) = dimnames(out)[[2]]
# plot(threshold, metrics["BalancedAccuracy",])
optThreshold_MCLUSTDA = threshold[which.max(metrics["BalancedAccuracy",])]
pred = predict(MCLUSTDA1, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[11,] = evalMetrics(data_test$y, prob_pred, optThreshold_MCLUSTDA)

# fit MCLUSTDA model on downsampled data
MCLUSTDA2 = MclustDA(data_downSamp[,1:11], data_downSamp$quality, modelType = "MclustDA")
pred = predict(MCLUSTDA2, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[12,] = evalMetrics(data_test$y, prob_pred)

# fit MCLUSTDA model on upsampled data
MCLUSTDA3 = MclustDA(data_upSamp[,1:11], data_upSamp$quality, modelType = "MclustDA")
pred = predict(MCLUSTDA3, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[13,] = evalMetrics(data_test$y, prob_pred)

# fit MCLUSTDA model on smote data
MCLUSTDA4 = MclustDA(data_smote[,1:11], data_smote$quality, modelType = "MclustDA")
pred = predict(MCLUSTDA4, newdata = data_test[,1:11])
prob_pred = pred$z[,"High"]
out[14,] = evalMetrics(data_test$y, prob_pred)

out[,1:4]
out[,5:8]
out[,9:12]


# Hypothyroid data example ---------------------------------------------

data = read.csv("hypothyroid.csv", header= TRUE, na.strings = "?", comment.char = "#")
data$class = factor(data$diagnosis, levels = c("negative", "hypothyroid"))
data = data[,c("age", "TSH", "T3", "TT4", "T4U", "FTI", "class")]
data = data[complete.cases(data),]
rownames(data) = NULL
tab = table(data$class)
cbind(Count = tab, "%" = prop.table(tab)*100)

set.seed(1)
train = caret::createDataPartition(1:nrow(data), p = 2/3)[[1]]
# training dataset
data_train = data[train,]
# test dataset
data_test = data[-train,]
data_test$y = ifelse(data_test$class == "hypothyroid", 1, 0)

# Imbalance Ratio
p = prop.table(table(data_train$class))
IR = max(p)/min(p)
IR

# Sampling
data_downSamp = caret::downSample(x = data_train[,1:6], 
                                  y = data_train$class)
names(data_downSamp)[ncol(data_downSamp)] = "class"
#
data_upSamp = caret::upSample(x = data_train[,1:6], 
                              y = data_train$class)
names(data_upSamp)[ncol(data_upSamp)] = "class"
#
data_smote = themis::smote(df = data_train[,1:7], 
                           var = "class")

out = array(as.double(NA), dim = c(14, 13))
dimnames(out) = list(c("EDDA", "EDDA+costSens",
                       "EDDA+adjPriorThreshold",
                       "EDDDA+optThreshold",
                       "EDDA+downSamp", "EDDA+upSamp", 
                       "EDDA+smote",
                       "MCLUSTDA", "MCLUSTDA+costSens",
                       "MCLUSTDA+adjPriorThreshold",
                       "MCLUSTDA+optThreshold",
                       "MCLUSTDA+downSamp", "MCLUSTDA+upSamp", 
                       "MCLUSTDA+smote"),
                     c("CE", "kappa", "logLoss", "Brier",
                       "Precision", "Recall", "F1", "F2",
                       "Sensitivity/TPR", "Specificity/TNR", 
                       "BalancedAccuracy", "YoudenScore", "AUC"))

# fit EDDA model on unbalanced data
EDDA1 = MclustDA(data_train[,1:6], data_train$class, modelType = "EDDA")
pred = predict(EDDA1, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[1,] = evalMetrics(data_test$y, prob_pred)

# fit EDDA model on unbalanced data + cost senstive predictions
pred = predict(EDDA1, newdata = data_test[,1:6], 
               prop = EDDA1$prop * c(1, IR))
prob_pred = pred$z[,"hypothyroid"]
out[2,] = evalMetrics(data_test$y, prob_pred)

# fit EDDA model on unbalanced data + threshold by adj prior prob
adjPriorProbs = classPriorProbs(EDDA1, data_train[,1:6])
pred = predict(EDDA1, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[3,] = evalMetrics(data_test$y, prob_pred, adjPriorProbs["hypothyroid"])

# fit EDDA model on unbalanced data + optimal threshold by CV
threshold = seq(0, 1, by = 0.01)
cv = cvMclustDA(EDDA1)
prob_cv = cv$z[,"hypothyroid"]
metrics = matrix(as.double(NA), nrow = 13, ncol = length(threshold))
for(i in 1:length(threshold))
{ 
  metrics[,i] = evalMetrics(ifelse(data_train$class == "hypothyroid", 1, 0), 
                            prob_cv, threshold[i]) 
}
rownames(metrics) = dimnames(out)[[2]]
# plot(threshold, metrics["BalancedAccuracy",])
optThreshold = threshold[which.max(metrics["BalancedAccuracy",])]
pred = predict(EDDA1, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[4,] = evalMetrics(data_test$y, prob_pred, optThreshold)

# fit EDDA model on downsampled data
EDDA2 = MclustDA(data_downSamp[,1:6], data_downSamp$class, modelType = "EDDA")
pred = predict(EDDA2, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[5,] = evalMetrics(data_test$y, prob_pred)

# fit EDDA model on upsampled data
EDDA3 = MclustDA(data_upSamp[,1:6], data_upSamp$class, modelType = "EDDA")
pred = predict(EDDA3, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[6,] = evalMetrics(data_test$y, prob_pred)

# fit EDDA model on smote data
EDDA4 = MclustDA(data_smote[,1:6], data_smote$class, modelType = "EDDA")
pred = predict(EDDA4, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[7,] = evalMetrics(data_test$y, prob_pred)

# fit MCLUSTDA model
MCLUSTDA1 = MclustDA(data_train[,1:6], data_train$class, modelType = "MclustDA", G = 1:5)
pred = predict(MCLUSTDA1, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[8,] = evalMetrics(data_test$y, prob_pred)

# fit MCLUSTDA model on unbalanced data + cost sensitive predictions
pred = predict(MCLUSTDA1, newdata = data_test[,1:6],
               prop = MCLUSTDA1$prop * c(1,IR))
prob_pred = pred$z[,"hypothyroid"]
out[9,] = evalMetrics(data_test$y, prob_pred)

# fit MCLUSTDA model on unbalanced data + threshold by adj prior prob
adjPriorProbs = classPriorProbs(MCLUSTDA1, data_train[,1:6])
pred = predict(MCLUSTDA1, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[10,] = evalMetrics(data_test$y, prob_pred, adjPriorProbs["hypothyroid"])

# fit MCLUSTDA model on unbalanced data + optimal threshold by CV
threshold = seq(0, 1, by = 0.01)
cv = cvMclustDA(MCLUSTDA1)
prob_cv = cv$z[,"hypothyroid"]
metrics = matrix(as.double(NA), nrow = 13, ncol = length(threshold))
for(i in 1:length(threshold))
{ 
  metrics[,i] = evalMetrics(ifelse(data_train$class == "hypothyroid", 1, 0), 
                            prob_cv, threshold[i]) 
}
rownames(metrics) = dimnames(out)[[2]]
# plot(threshold, metrics["BalancedAccuracy",])
optThreshold = threshold[which.max(metrics["BalancedAccuracy",])]
pred = predict(MCLUSTDA1, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[11,] = evalMetrics(data_test$y, prob_pred, optThreshold)

# fit MCLUSTDA model on downsampled data
MCLUSTDA2 = MclustDA(data_downSamp[,1:6], data_downSamp$class, modelType = "MclustDA")
pred = predict(MCLUSTDA2, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[12,] = evalMetrics(data_test$y, prob_pred)

# fit MCLUSTDA model on upsampled data
MCLUSTDA3 = MclustDA(data_upSamp[,1:6], data_upSamp$class, modelType = "MclustDA")
pred = predict(MCLUSTDA3, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[13,] = evalMetrics(data_test$y, prob_pred)

# fit MCLUSTDA model on smote data
MCLUSTDA4 = MclustDA(data_smote[,1:6], data_smote$class, modelType = "MclustDA")
pred = predict(MCLUSTDA4, newdata = data_test[,1:6])
prob_pred = pred$z[,"hypothyroid"]
out[14,] = evalMetrics(data_test$y, prob_pred)
  
out[,1:4]
out[,5:8]
out[,9:12]

