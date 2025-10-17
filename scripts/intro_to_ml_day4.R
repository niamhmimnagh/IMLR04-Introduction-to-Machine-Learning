# K-MEANS CLUSTERING
library(tidymodels)

# Iris dataset

set.seed(123)

# Import the iris dataset
data(iris)
head(iris)
iris<-as.tibble(iris)

# RECIPE
# unsupervised learning - removed the species
# we dont have y (outcome/response), but we have all of our predictors
# outcome ~ .
km_rec<-recipe(~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
               data=iris) %>%
  step_normalize(all_numeric_predictors())

# MODEL SPECIFICATION
scan_k<-function(k){
  spec_k<-k_means(num_clusters = k) %>%
    set_engine("stats")
  
  wf_k<-workflow() %>%
    add_recipe(km_rec) %>%
    add_model(spec_k)
  
  fit_k<-fit(wf_k, data=iris)
  wss<-workflows::pull_workflow_fit(fit_k)$fit$tot.withinss # extract the k-means engine
  # from the fitted workflow, and read out its estimated within cluster sum of squares
  tibble(k=k, wss=wss) # creates a dataset - with the k value and within cluster sum of squares
}


# Decide how many values of k we want to look at
# check k=1...10 - fit k-means clustering for 1 cluster, 2 clusters, 3 clusters ... 10 clusters
# here map_dfr calls scan_k function for k=1, k=2, ..., k=10
elbow_tbl<-map_dfr(1:10, scan_k)
head(elbow_tbl, 10)

# Plot the elbow plot
# ggplot - you can specify if you want a scatter plot or a line plot
library(ggplot2)
ggplot(elbow_tbl, aes(x=k, y=wss)) +
  geom_line()+
  geom_point()


# K=4
k_opt<-4
# model spec
spec_k<-k_means(num_clusters = 4) %>%
  set_engine("stats")

# workflow
wf_k<-workflow() %>%
  add_recipe(km_rec) %>%
  add_model(spec_k)
# fit model
fit_k<-fit(wf_k, data=iris)


# assign each observation to its cluster
# augment is used to do that

iris_aug<-augment(fit_k, new_data=iris)

head(iris_aug)



# Validating our clusters
# Adjusted Rand Index (AR1), NMI

library(mclust) # ARI
library(aricode) #NMI


# ARI
mclust::adjustedRandIndex(iris$Species, iris_aug$.pred_cluster)
# ARI goes from -1 to 1 with 1 being perfect agreement
# so 0.55 moderate agreement with the true values

# NMI
aricode::NMI(iris$Species, iris_aug$.pred_cluster)
# moderate agreement

library(cluster)
# silhouette scores
#cluster::silhouette(iris_aug$.pred_cluster)
# DB 

# Hierarchical Clustering

set.seed(123)

# recipe - removed the 'Species' response column and it normalised all of our predictors
hc_rec<-recipe(~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
               data=iris) %>%
  step_normalize(all_numeric_predictors())

# model specification
# options include 'single' (closest), 'complete' (furthest), 'average', 'ward.D2'
# we dont specify set_mode() - in unsupervised learning
hc_spec<-hier_clust(linkage_method = 'ward.D2') %>%
  set_engine('stats')


# workflow - combine the recipe and the model spec

hc_wf<-workflow() %>%
  add_recipe(hc_rec) %>%
  add_model(hc_spec)

# Fit the model
hc_fit<-fit(hc_wf, data=iris)

# Dendrogram
library(tidyclust)
dendrogram<-tidyclust::extract_fit_engine(workflows::pull_workflow_fit(hc_fit))

# Plot the dendrogram
library(factoextra)
fviz_dend(dendrogram)

# k=3 - num_clusters=3
assign_tbl<-tidyclust::extract_cluster_assignment(workflows::pull_workflow_fit(hc_fit), num_clusters=3)
# hc_fit is workflow (recipe and our model)
# pull_workflow_fit opens up the workflow and just extracts the fitted model object
# extract_cluster_assignment expects the model fit, not the whole workflow 

assign_tbl<-tidyclust::extract_cluster_assignment(dendrogram, num_clusters=3) # column with cluster assignment

fviz_dend(dendrogram, k=4)


iris_hc_aug<-iris %>% bind_cols(assign_tbl %>% transmute(hc_cluster = .cluster))
head(iris_hc_aug)

mclust::adjustedRandIndex(iris_hc_aug$Species, iris_hc_aug$hc_cluster)
# still moderate, but an improvement over k-means

aricode::NMI(iris_hc_aug$Species, iris_hc_aug$hc_cluster)
# an improvement over k-means

# Internal validation metrics 
#  Get the preprocessed predictors used for clustering
rec_prep <- prep(hc_rec, training = iris)
X <- bake(rec_prep, new_data = iris)           
D <- dist(X, method = "euclidean")             

# Silhouette 
sil_obj <- cluster::silhouette(as.integer(iris_hc_aug$hc_cluster), D)
mean(sil_obj[,3])
# <0.25 is poor
# 0.25-0.5 moderate
# 0.5-0.7 good
# >0.7 strong
# clusters are moderately separable according to the silhouette score

# Davies–Bouldin index 
X_mat <- as.matrix(X)
db_out <- clusterCrit::intCriteria(X_mat,
                                   part = as.integer(iris_hc_aug$hc_cluster),
                                   crit = "Davies_Bouldin")
db_out$davies_bouldin
# lower DB score is better - 
# < 0.5 excellent
# 0.5 - 1 is good
# greater is weak

# decent but not perfect clustering

# PCA on wine data

library(rattle)
data(wine, package='rattle')
head(wine)
wine_pca<-prcomp(wine[,-1], center=TRUE, scale=TRUE)


# Decide on number of principal components - scree plot
fviz_eig(wine_pca, addlabels=TRUE)
# the first 5 account for roughly 80% of the variance

# Biplot
fviz_pca_biplot(wine_pca)

# Whats going on with the others?
wine_pca$rotation
# PC1 - higher weights for flavanoids, phenols and dilution
# a wine that is negative in terms of pc1 - higher in dilution, hue, phenols, flavanoids - richness/body characteristics
# a wine that is positive in terms of pc1 - higher in nonflavanoids, alcalinity
# pc1 broadly separates wines in terms of phenols and richness

# pcc2
# a wine that is negative in pc2 would have high alcohol and a dark colour
#  a wine that is positive in pc2 would have higher dilution, higher hue 

