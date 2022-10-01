rm(list =ls())
library(httr)
library(data.table)
library(ggplot2)
library(caret)
library(ClusterR)
library(Metrics)

dat_test <- fread('./project/volume/data/interim/dat_test_melt.csv')
dat_train <- fread('./project/volume/data/interim/dat_train_melt.csv')

# making an identifier for the ID's so that it's easy to identify them when 
# splitting the data back to train and test
dat_train$id_identifier <- 0
dat_test$id_identifier <- 1
# saving the labels, train id, and test id before getting rid of them to 
# perform PCA
dat_train_labels <- dat_train$labels_SR
dat_train$labels_SR <- NULL
dat_train_id <- dat_train$id
dat_test_id <- dat_test$id
dat_train$id <- NULL
dat_test$id <- NULL

# merging the train and test data table so that we perform PCA so that we avoid 
# the errors caused by rotation 
dat_merged <- rbind(dat_train, dat_test)

dat_merged_id_identifier <- dat_merged$id_identifier
dat_merged$id_identifier <- NULL

# performing PCA
pca <- prcomp(dat_merged)

# converting the PCA's back into a data table and unclassing it 
dat_pca <- data.table(unclass(pca)$x)

# putting back the identifier variable into the data table
dat_pca$id_identifier <- dat_merged_id_identifier

fwrite(dat_pca, "./project/volume/data/interim/dat_pca.csv")
