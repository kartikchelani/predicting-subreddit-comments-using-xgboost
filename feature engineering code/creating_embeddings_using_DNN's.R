rm(list = ls())
library(httr)
library(data.table)

getEmbeddings <- function(text){ 
  input <- list(
    instances = list( text)
  )
  res <- POST("https://dsalpha.vmhost.psu.edu/api/use/v1/models/use:predict", body = input,encode = "json", verbose())
  emb <- unlist(content(res)$predictions)
  emb
}

dat_test <- fread('./project/volume/data/interim/dat_test.csv')
dat_train <- fread('./project/volume/data/interim/dat_train.csv')

# make new empty data tables so that I can use it to rbind my embeddings 
emb_dt_train <- NULL
emb_dt_test <- NULL

# for loop that iterates through all the "text" rows in train and test to 
# covert it into 512 vectors
for (i in 1:nrow(dat_train)) {
  emb_dt_train <- rbind(emb_dt_train, getEmbeddings(dat_train$text[i]))
}
emb_dt_train <- data.table(emb_dt_train)

for (i in 1:nrow(dat_test)) {
  emb_dt_test <- rbind(emb_dt_test, getEmbeddings(dat_test$text[i]))
}
emb_dt_test <- data.table(emb_dt_test)

fwrite(emb_dt_train, "./project/volume/data/interim/embedded train.csv")
fwrite(emb_dt_test, "./project/volume/data/interim/embedded test.csv")