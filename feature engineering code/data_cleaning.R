rm(list = ls())
library(data.table)

# reading in the data
dat_train <- fread("./project/volume/data/raw/train_data.csv")
dat_test <- fread("./project/volume/data/raw/test_data.csv")

# getting rid of empty strings in case there are any
dat_train <- dat_train[!dat_train[,text == ""]]
dat_test <- dat_test[!dat_test[,text == ""]]

fwrite(dat_train,"./project/volume/data/interim/dat_train.csv")
fwrite(dat_test,"./project/volume/data/interim/dat_test.csv")