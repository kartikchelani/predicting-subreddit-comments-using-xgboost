rm(list =ls())
library(data.table)
library(ggplot2)
library(caret)
library(ClusterR)
library(Metrics)
library(Rtsne)

set.seed(9001)

dat_test <- fread('./project/volume/data/interim/dat_test_melt.csv')
dat_train <- fread('./project/volume/data/interim/dat_train_melt.csv')
dat_pca <- fread("./project/volume/data/interim/dat_pca.csv")

# saving labels, identifier, and the id's from the melted dataset to put it 
# into the tsne data table after completing tsne
dat_train_labels <- dat_train$labels_SR
dat_train_id <- dat_train$id
dat_test_id <- dat_test$id
dat_train_id_identifier <- dat_pca$id_identifier
dat_pca$id_identifier <- NULL

# performing tsne's with different perplexities 
tsne_dat <- Rtsne(dat_pca, # pca data table
                  dims = 2,
                  pca=F, 
                  max_iter=2000,
                  perplexity = 20,
                  theta = 0.3,
                  check_duplicates = F)
tsne_dat_1 <-  Rtsne(dat_pca, 
                     dims = 2,
                     theta = 0.3,
                     pca=F, #F
                     max_iter=2000,
                     perplexity = 30,
                     check_duplicates = F)
tsne_dat_2 <- Rtsne(dat_pca, 
                    dims = 2,
                    pca=F, 
                    max_iter=2000,
                    perplexity = 50,
                    theta = 0.3,
                    check_duplicates = F)
tsne_dat_3 <- Rtsne(dat_pca, 
                    dims = 2,
                    pca=F, 
                    max_iter=2000,
                    perplexity = 70,
                    theta = 0.3,
                    check_duplicates = F)
tsne_dat_4 <- Rtsne(dat_pca, 
                    dims = 2,
                    pca=F, 
                    max_iter=2000,
                    perplexity = 90,
                    theta = 0.3,
                    check_duplicates = F)

# converting all tsne's intoa data table and changing their names to avoid 
# conflicting names
tsne_data_table <- data.table(tsne_dat$Y)

tsne_data_table_1 <- data.table(tsne_dat_1$Y)
tsne_data_table_1 <- setnames(tsne_data_table_1,"V1","V3")
tsne_data_table_1 <- setnames(tsne_data_table_1, "V2", "V4")

tsne_data_table_2 <- data.table(tsne_dat_2$Y)
tsne_data_table_2 <- setnames(tsne_data_table_2, "V1", "V5")
tsne_data_table_2 <- setnames(tsne_data_table_2,"V2","V6")

tsne_data_table_3 <- data.table(tsne_dat_3$Y)
tsne_data_table_3 <- setnames(tsne_data_table_3, "V1", "V7")
tsne_data_table_3 <- setnames(tsne_data_table_3, "V2", "V8")

tsne_data_table_4 <- data.table(tsne_dat_4$Y)
tsne_data_table_4 <- setnames(tsne_data_table_4, "V1", "V9")
tsne_data_table_4 <- setnames(tsne_data_table_4, "V2", "V10")

# cbind all tsne data tables
tsne_combined <- cbind(tsne_data_table, tsne_data_table_1, tsne_data_table_2, tsne_data_table_3, tsne_data_table_4)
tsne_combined$id_identifier <- dat_train_id_identifier 

# subsetting my data back into train and test using the idenfitfier 
train <- subset(tsne_combined, id_identifier==0)
test <- subset(tsne_combined, id_identifier==1)

# assigning the id's and labels into their respective data tables
train$id <- dat_train_id
test$id <- dat_test_id
train$labels_SR <- dat_train_labels

# getting rid of the identifiers because they have served their purpose
train$id_identifier <- NULL
test$id_identifier <- NULL

fwrite(train, "./project/volume/data/interim/train_xgboost.csv")
fwrite(test, "./project/volume/data/interim/test_xgboost.csv")
