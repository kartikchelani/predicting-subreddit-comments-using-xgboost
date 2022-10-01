rm(list = ls())
library(httr)
library(data.table)

test <- fread('./project/volume/data/raw/test_data.csv')
train <- fread('./project/volume/data/raw/train_data.csv')
dat_test <- fread('./project/volume/data/interim/embedded test.csv')
dat_train <- fread('./project/volume/data/interim/embedded train.csv')

# extracting the id variable from the main train and test dataset
# I forgot to to this in the embeddings file so I'm doing it here instead
test_id <- test$id
train_id <- train$id

# assigning the the id's to the data tables on which I'm going to perform melt()
dat_test$id <- test_id
dat_train$id <- train_id

# using the melt() for all columns excluding the text because that's not needed
melt_train <- train[,!c("text")]
labels_SR_train <- melt(melt_train, id.vars='id')

# selecting the relevant subreddits for the id's
labels_SR_train <- labels_SR_train[value=='1']

# using as.numeric() to assign them their labels
labels_SR_train$labels_SR <- as.numeric(labels_SR_train$variable)-1
labels_SR_train$value <- NULL

# merging the data.tables by id
dat_train <- merge.data.table(dat_train,labels_SR_train, by='id')
dat_train$variable <- NULL

fwrite(dat_train, "./project/volume/data/interim/dat_train_melt.csv")
fwrite(dat_test, "./project/volume/data/interim/dat_test_melt.csv")