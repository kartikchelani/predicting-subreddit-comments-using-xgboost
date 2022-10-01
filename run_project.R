# final project run project guide
# feature engineering 
source('./project/src/features/clean data.R')
source('./project/src/features/embeddings.R')
source('./project/src/features/melt.R')

# running xgboost before pca and tsne
source('./project/src/models/xgboost.R')

# feature engineering (pca and tsne)
source('./project/src/features/pca.R')
source('./project/src/features/tsne.R')

# xgboost after performing pca and tsne
source('./project/src/models/xgboost_after_tsne.R')