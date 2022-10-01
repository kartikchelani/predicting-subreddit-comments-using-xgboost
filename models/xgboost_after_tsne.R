# clear work environment and load packages needed
# This model uses xgboost after performing PCA and TSNE
rm(list =ls())
library(httr)
library(data.table)
library(caret)
library(Metrics)
library(xgboost)

set.seed(9001)

# reading in the tsne data tables from the tsne.r feature file
test <- fread('./project/volume/data/interim/test_xgboost.csv')
train <- fread('./project/volume/data/interim/train_xgboost.csv')

# saving the test and train id and 
# assigning the label to the train variable
train_id <- train$id
test_id <- test$id
y.train <- train$labels_SR
# deleting the id and labels column from the data tables 
train$id <- NULL
test$id <- NULL
train$labels_SR <- NULL

# converting the test and train data table to a matrix form
train_matrix <- as.matrix(train)
test_matrix <- as.matrix(test)

hyper_perm_tune <- NULL

train_xgb <- xgb.DMatrix(train_matrix, label=y.train, missing = NA)
test_xgb <- xgb.DMatrix(test_matrix, missing = NA)

# using a nested for loop to search for the best combination of parameters
for (best_tree_depth in c(1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60)) {
    for (best_eta in c(0.01, 0.08, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5)) {

      param <- list(  objective           = "multi:softprob",
                      num_class = 10,
                      gamma               = 0.02,   # minimum loss reduction 
                      booster             = "gbtree",
                      eval_metric         = "mlogloss",
                      eta                 = best_eta,    # default eta of 0.3
                      max_depth           = best_tree_depth,      
                      subsample           = 0.9,
                      colsample_bytree    = 1.0,    
                      tree_method = 'hist'
      )

      XGBfit <- xgb.cv(params = param,
                      nfold = 5,  
                      nrounds = 3000,
                      missing = NA,
                      data = train_xgb,
                      print_every_n = 2,
                      early_stopping_rounds = 30)   # training stops when there is no improvement

      best_tree_n <- unclass(XGBfit)$best_iteration # extract best iteration 
    
      new_row <- data.table(t(param))
      new_row$best_tree_n <- best_tree_n
      # calculates the mean log loss test error and adds it to a data table to
      # help us choose the best combination
      test_error <- unclass(XGBfit)$evaluation_log[best_tree_n,]$test_mlogloss_mean
      new_row$test_error <- test_error
      hyper_perm_tune <- rbind(new_row, hyper_perm_tune)
}}

# using the best tuning parameters from the step above

best_parameters <-  list(  objective           = "multi:softprob",
                           num_class = 10,
                           gamma               = 0.02,   # minimum loss reduction 
                           booster             = "gbtree",
                           eval_metric         = "mlogloss",
                           eta                 = 0.35, #0.4   # default eta of 0.3
                           max_depth           = 15,  #25    
                           subsample           = 0.9,
                           colsample_bytree    = 1.0,    
                           tree_method = 'hist'
)


watchlist <- list( train = train_xgb)



XGBfit <- xgb.train( params = best_parameters,
                     nrounds = best_tree_n,
                     missing = NA,
                     data = train_xgb,
                     watchlist = watchlist,
                     print_every_n = 1)

# predicting  which subreddit the text belongs to on the test data
pred <- predict(XGBfit, newdata = test_xgb)

# making the submission file below, renaming it to appropriate columns, 
# and setting the column order for final submission 

submission <- NULL
submission  <-  data.table(submission)
# converting 205540 preds from one column into 10 columns to get the file in 
#the format needed
submission <- data.frame(matrix(unlist(pred), nrow=20554, byrow=TRUE),stringsAsFactors=TRUE)
submission$id <- test_id

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

fwrite(submission, "./project/volume/data/processed/submission_xgboost_tsne_max_params.csv")
