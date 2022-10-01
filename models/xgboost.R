# This model uses xgboost before performing PCA and TSNE
rm(list =ls())
library(httr)
library(data.table)
library(caret)
library(Metrics)
library(xgboost)

set.seed(9001)

main_test <- fread('./project/volume/data/raw/test_data.csv')
main_train <- fread('./project/volume/data/raw/train_data.csv')
dat_test <- fread('./project/volume/data/interim/dat_test_melt.csv')
dat_train <- fread('./project/volume/data/interim/dat_train_melt.csv')

hyper_perm_tune <- NULL

y.train <- dat_train$labels_SR

dat_train_matrix <- as.matrix(dat_train[,2:513])
dat_test_matrix <- as.matrix(dat_test[,1:512])

train <- xgb.DMatrix(dat_train_matrix, label=y.train, missing = NA)
test <- xgb.DMatrix(dat_test_matrix, missing = NA)

param <- list(  objective           = "multi:softprob",
                num_class = 10,
                gamma               = 0.02,   # minimum loss reduction 
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                eta                 = 0.15, #*i   # default eta of 0.3
                max_depth           = 8,      
                subsample           = 0.9,
                colsample_bytree    = 1.0,    
                tree_method = 'hist'
)

XGBfit <- xgb.cv(params = param,
                 nfold = 5,  #5fold CV
                 nrounds = 10000, # maximum num of trees
                 missing = NA,
                 data = train,
                 print_every_n = 2,
                 early_stopping_rounds = 30)   # training stops when no improvement

best_tree_n <- unclass(XGBfit)$best_iteration
new_row <- data.table(t(param))
new_row$best_tree_n <- best_tree_n

test_error <- unclass(XGBfit)$evaluation_log[best_tree_n,]$test_rmse_mean
new_row$test_error <- test_error
hyper_perm_tune <- rbind(new_row, hyper_perm_tune)

watchlist <- list(train = train)


XGBfit <- xgb.train( params = param,
                     nrounds = best_tree_n,
                     missing = NA,
                     data = train,
                     watchlist = watchlist,
                     print_every_n = 1)

pred <- predict(XGBfit, newdata = test)
submission <- NULL
submission  <-  data.table(submission)
submission$id <- main_test$id
submission <- data.frame(matrix(unlist(pred), nrow=20554, byrow=TRUE),stringsAsFactors=TRUE)
submission$id <- main_test$id

submission <- setnames(submission,  "X1", "subredditcars")
submission <- setnames(submission,  "X2", "subredditCooking")
submission <- setnames(submission,  "X3", "subredditMachineLearning")
submission <- setnames(submission,  "X4", "subredditmagicTCG")
submission <- setnames(submission,  "X5", "subredditpolitics")
submission <- setnames(submission,  "X6", "subredditReal_Estate")
submission <- setnames(submission,  "X7", "subredditscience")
submission <- setnames(submission,  "X8", "subredditStockMarket")
submission <- setnames(submission,  "X9", "subreddittravel")
submission <- setnames(submission,  "X10", "subredditvideogames")
submission <- setcolorder(submission, c(11,1,2,3,4,5,6,7,8,9,10))

fwrite(submission, "./project/volume/data/processed/submission.csv")
