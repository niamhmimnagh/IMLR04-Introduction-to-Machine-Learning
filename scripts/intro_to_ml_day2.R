# Introduction to Machine Learning Day 2
# Linear Regression, Regularised Regression, Random Forest
library(tidymodels)
library(modeldata)

# Import data
data(ames,package="modeldata")

glimpse(ames)

# IN PRACTICE - perform your EDA: looks at your visualisations, missing data, relationship, correlations

skimr::skim(ames)


# SPLIT YOUR DATA BEFORE PREPROCESSING
library(rsample)
# specify your dataset, the proportion you want to save for training/testing, and your response
# stratification is useful for classification,  not for regression
split<-initial_split(ames, prop=0.7)
train<-training(split)
test<-testing(split)

# PREPROCESSING STEPS:

# 1. impute any missing values
# 2. collapse any low-frequency categories into a 'other' column
# 3. encode any of our categorical variables
# 4. drop any zero-variance variables (variables with the same value all the way through)
# 5. normalise our numeric predictors

library(recipes)

# Building the recipe:
# Sale_Price ~ . means: we want to model sale price using all of the predictors
house_recipe<-recipe(Sale_Price ~ . , data=train) %>% # use the training set, not the full one
  step_log(Sale_Price, skip=TRUE) %>% # skip=true means only do this to the training data, not the test
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>% # 'nominal' instead of 'categorical'
  step_other(all_nominal_predictors(), threshold=0.01) %>% # if a level is less than 1% of the dataset, collapset it into 'other' category
  step_mutate(Mo_Sold=factor(Mo_Sold)) %>% # turning the 'month sold' column into a factor
  step_dummy(all_nominal_predictors()) %>%  # encoding our nominal predictors as 0/1 columns
  step_zv(all_predictors()) %>% # remove zero-variance predictors
  step_corr(all_numeric_predictors(), threshold=0.95) %>%
  step_normalize(all_numeric_predictors()) # standardise numeric predictor

# Model Specification:
lm_spec<-linear_reg() %>%
  set_engine("lm")


# Workflow (order of model spec and recipe are not important):
workflow_lm<-workflow() %>%
  add_model(lm_spec)%>%
  add_recipe(house_recipe)

# Fit the model to the training data:
fit_lm<- fit(workflow_lm, data=train)
tidy(fit_lm) # extract coefficients using the tidy function

# Predict on test data
# get predictions on the log scale for the test set
pred_log <- predict(fit_lm, new_data = test)

#  back-transform predictions to the original  
# to convert from log-price -> price, we use the exp()
pred_orig <- pred_log %>%
  dplyr::mutate(Sale_price_pred = exp(.pred)) %>%
  dplyr::select(Sale_price_pred)

# combine truth (test$Sale_Price) with the back-transformed predictions
test_results <- test %>%
  dplyr::select(Sale_Price) %>%
  dplyr::bind_cols(pred_orig)

# compute test-set metrics on the original scale
metrics(test_results, truth = Sale_Price, estimate = Sale_price_pred)


# RMSE  measures the average size of the prediction error
# on average the model predictions differ from the true sale price by about 26k
min(test$Sale_Price)
max(test$Sale_Price)

# MAE is the average absolute difference between the predicted and actual prices
# on average the model is off by about 16.7k for each house

# R2 is 91.2% - about 91% of the variation in sale price is explained by our model predictors
# its a measure of how well your model captures the meaningful structure in the data
# R2 = 1-(sum(observed Y - predicted Y)^2)/(observed Y - mean(Y))^2
# 1 - SSE/SST

# PLOT RESULTS
library(ggplot2)
ggplot(test_results, aes(x=Sale_Price, y=Sale_price_pred))+
  geom_point()

# REGULARISED REGRESSION - ELASTIC NET
# penalty - lambda
# mixture - alpha

# split our data
# specify our recipe
# specify our model specification

enet_spec<- linear_reg(
  penalty = tune(), # lambda is the overal regularisation strength
  mixture = tune(), # alpha = 0 - ridge regression, alpha=1 - lasso, anything in between - elastic net
) %>% 
  set_engine("glmnet")

# Workflow

workflow_enet<- workflow() %>%
  add_model(enet_spec)  %>% 
  add_recipe(house_recipe)


# Resampling + Grid
set.seed(123)

# k folds cross validation, with k=v
folds<-vfold_cv(train, v=5)  # split the training data into 5 sections (folds), use 4 to train and keep 1 for validation

# Search a reasonable space of (penalty, mixture)
enet_grid<-grid_latin_hypercube(
  penalty(),       # penalty(range=c(0.1,0.5)) default: 1e-4 -> 1
  mixture(),       # default 0 - 1
  size = 10
)

# Choosing our metrics
enet_metrics<- metric_set(rmse, rsq, mae)  

# Tuning our model  
tuned_enet<-tune_grid(
  workflow_enet,
  resamples = folds, # resampling folds to be used for model tuning
  grid = enet_grid,
  metrics = enet_metrics,
  control = control_grid(save_pred =  TRUE) # save predictions across cross-validation
)
  
# Pick the best hyperparameters by RMSE - you could also choose R2 or MAE
best_enet<-select_best(tuned_enet, metric="rmse")
best_enet

# Once we have our optimum hyperparameters, we lock in the best parameters in our workflow
# and then we refit on the full training data

final_enet_wf<-finalize_workflow(workflow_enet, best_enet)

# Finally we fit
fit_enet<-fit(final_enet_wf, data=train)
tidy(fit_enet) # look at your coefficients

# PREDICT
# Predict on test data
# get predictions on the log scale for the test set
pred_log <- predict(fit_enet, new_data = test)

# #  back-transform predictions to the original  
# # to convert from log-price -> price, we use the exp()
# pred_orig <- pred_log %>%
#   dplyr::mutate(Sale_price_pred = exp(.pred)) %>%
#   dplyr::select(Sale_price_pred)
# 
# # combine truth (test$Sale_Price) with the back-transformed predictions
# test_results <- test %>%
#   dplyr::select(Sale_Price) %>%
#   dplyr::bind_cols(pred_orig)


# RANDOM FOREST
# we have our data split into training/testing and we have our recipe

# we need our model specification 
rf_spec<-rand_forest(
  mtry =  tune(), 
  trees = 500,
  min_n = 5
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Workflow - recipe + model specification

workflow_rf<-workflow() %>%
  add_model(rf_spec)%>%
  add_recipe(house_recipe)


# Specify your cross validation after your training and testing split
# we need to give a range for mtry because it needs to know how many parameters are available for splitting
# we run our preprocessing on the training data so that we know how many parameters will be available for
# selection in the model after the preprocessing takes place
rec_prep<-prep(house_recipe, training=train) # this returns the values that are needed for preprocessing to take place
a<-bake(rec_prep, new_data=train)
ncol(a) # how many columns do we have? 200: 200 - sale price  = 199 variables

rf_grid<-grid_latin_hypercube(
  mtry(range = c(1, 199)), # (10, 100)
  size=10
)

# Tune via cross validation
rf_results<-tune_grid(
  workflow_rf, 
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(rmse, rsq, mae)
)


# PICK THE BEST MTRY VALUE
best_rf<-select_best(rf_results, metric="mae")

# Finalise the workflow and fit the model to the entire training dataset
final_rf<-finalize_workflow(workflow_rf, best_rf)
fit_rf<-fit(final_rf, data=train)



