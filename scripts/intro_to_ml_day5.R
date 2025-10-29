# USE AN SVM TO PREDICT SPAM/NOT SPAM

library(tidymodels)
library(DAAG)
library(vip) # variable importance
library(rsample)
# Load in the dataset
data('spam7', package="DAAG")
head(spam7) 

# Response needs to be a factor in order for stratified sampling
str(spam7) # yesno is a factor with 2 levels

# If it wasn't:
spam7$yesno<-factor(spam7$yesno, levels=c('n', 'y'))

# Stratified random sampling into test and train datasets
spam_split<-initial_split(spam7, prop=0.7, strata=yesno)
spam_train<-training(spam_split)
spam_test<-testing(spam_split)


# cross validation splits - stratifiying to keep our response in the same proportions as our overall dataset
spam_folds<-vfold_cv(spam_train, v=5, strata=yesno) 


# RECIPE - Preprocessing
spam_rec<-recipe(yesno ~ ., data=spam_train) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric())

# MODEL SPECIFICATION - RBF SVM with tuning
svm_spec<-svm_rbf(
  cost = tune(), # penalty for margin violations 
  rbf_sigma = tune() # kernel width
) %>%
  set_engine('kernlab') %>%
  set_mode('classification')

#  WORKFLOW - RECIPE + MODEL SPEC

svm_wf<- workflow() %>%
  add_model(svm_spec) %>%
  add_recipe(spam_rec)

# SET UP THE GRID FOR YOUR HYPERPARAMETERS - 
svm_grid<-grid_regular(
  cost(range=c(-3,3)), # cost works on the 2^cost
  rbf_sigma(range=c(-4,1)), 
  levels=5
)

svm_grid

# tidymodels documentation online will tell you the scale that the hyperparameters are functioning on

# Run cross validation to check hyperparameter values
svm_tuned<-tune_grid(
  svm_wf,
  resamples = spam_folds,
  grid = svm_grid, 
  metrics = metric_set(roc_auc, pr_auc, accuracy)
)

# SELECT THE BEST RESULTS
best_params<-select_best(svm_tuned, metric='roc_auc')
best_params # the best cost is 8 which suggests the model might benefit from examining larger costs
# the select_best function will just give you the hyperparameter combination with the largest roc_auc
# even if that combination is only marginally/very slightly better than a combination that might
# generalise better

# VIEW MODEL RESULTS OVER TUNING PROCESS
autoplot(svm_tuned)

# FINALISE YOUR WORKFLOW
second_best<-tibble('cost'=1, 'rbf_sigma' = 0.0316)
svm_final_wf<-finalize_workflow(svm_wf, second_best)

# FIT THE MODEL ON THE TRAINING SET
svm_fit<-fit(svm_final_wf, data=spam_train)

# PREDICT ON OUR TEST DATA
svm_preds<-predict(svm_fit, spam_test, type = 'prob') %>%
  bind_cols(predict(svm_fit, spam_test, type = 'class')) %>%
  bind_cols(spam_test %>% select(yesno))

head(svm_preds) # structure check

#  METRICS
# Accuracy
library(yardstick)
table(spam_test$yesno)/nrow(spam_test)

accuracy(
  data =  svm_preds, 
  truth =  yesno, 
  estimate=.pred_class
)

# accuracy of 84.5% - 84.5% of emails are being classified correctly


# CONFUSION MATRIX
conf_mat(
  data =  svm_preds, 
  truth =  yesno, 
  estimate=.pred_class
)

(781+386)/(781+386+158+56)

levels(spam_test$yesno)
# RECALL - sensitivity (ecology)
sens(
  data =  svm_preds, 
  truth =  yesno, 
  estimate=.pred_class,
  event_level = 'second' # we're interested in 'y' which is the second level
)

# Recall of 71%
# among true spam emails (yesno =  'y'), 71% are being correctly flagged as being spam
# miss-rate of 29% - slipping through as non-spam
# change the decision threshold (currently it is 0.5 > reducing it would increase the recall, resample your
# data before modelling)


library(vip)
set.seed(123) # set seed to make shuffles reproducible

system.time( # can be useful, sometimes is required to know how long your algorithm takes to run
  vi_perm<-vip::vi(
    object = svm_fit, # fitted tidymodels model
    method = 'permute', # permutation based importance (shuffle features)
    train = spam_train, 
    target = 'yesno',
    metric = 'roc_auc', # metric for determining variable importance
    event_level = "second",
    pred_wrapper = function(object, newdata){ # numeric P(event)
      p<-predict(object, new_data = newdata, type = 'prob')[[paste0(".pred_", "y")]]
      as.numeric(p)
    },
    nsim = 5,
    reference_class = "y",
    keep = TRUE # keep per-permutation results
  )
)
vip::vip(vi_perm, num_features = nrow(vi_perm))

# how important each variable was predicting spam- 
# the number of exclamation marks was the most important predictor of spam

# ALTERNATIVE CODE FOR CONTINUOUS OUTCOMES # 
# SHAP VALUES # 

##############
# NEURAL NETWORKS
# Predicting house prices from the Ames housing dataset using nnet for regression

# Import data
data(ames, package='modeldata')

# Split the data (before preprocessing to avoid leakage)
ames_split<-initial_split(ames, prop=0.7)
ames_train<-training(ames_split)
ames_test<-testing(ames_split)

# Create our cross-validation folds
ames_folds<-vfold_cv(ames_train, v=5)
  

# RECIPE
ames_rec<-recipe(Sale_Price ~ . , data=ames_train) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>% # step_nzv() remove near zero variance predictors
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold=0.9) %>% # removes numeric predictors with correlation > 0.9
  step_lincomb(all_predictors()) # drop numeric predictors that are a linear combination of others

# 000000000101011111111101 # dummy coding example
# 01110100000000001111100


# MODEL SPEC
nn_spec<- mlp(
  hidden_units=tune(),
  penalty = tune(),
  epochs = tune()
) %>%
  set_mode("regression") %>%
  set_engine('nnet', trace=FALSE, MaxNWts = 20000) # MaxNWts = greater than the number in the complaint output

# WORKFLOW
nn_wf<-workflow() %>%
  add_recipe(ames_rec) %>%
  add_model(nn_spec)


# Hyperparameter grid
# L means integer - R works with doubles by default
nn_grid<-grid_latin_hypercube(
  hidden_units(range=c(5L, 50L)), # number of neurons in the hidden layer - controls the model capacity
                # small values (5) - simple, low-capacity model (may underfit)
                # large values (50) - complex model (may overfit if not regularised)
                # more neurons means more weight parameters to estimate  -  can be slow
  penalty(range = c(-5, 0)),     # regularisation paraameter - sampled on the log10 scale
                #higher penalty shrinks the weights - smoother model and less overfitting
                # too high - underfitting, too low - underfitting
  epochs(range = c(50L, 300)), # epochs - number of training iterations - how many times the data will pass through the network
                # low epochs (50) and your network might stop too early - underfitting
                # high- allow for more convergence - less likely to underfit but you could risk overfitting
                # higher penalty (regularisation) - more epochs are safe
  size =  5
    
)
nn_grid # latin hypercube randomly samples from the range you give it 

# TUNE THE MODEL
nn_tuned<- tune_grid(
  nn_wf, 
  resamples = ames_folds,
  grid = nn_grid, 
  metrics = metric_set(rmse, rsq, mae)
)

# Extract the best hyperparameters
best_nn<-select_best(nn_tuned, metric='rmse')
autoplot(nn_tuned)

# We've chosen our optimum hyperparameters
# we need to fit our model on the entire training data
nn_final_wf<-finalize_workflow(nn_wf, best_nn)
nn_fit<-fit(nn_final_wf, data=ames_train)

# Predictions
nn_preds<-predict(nn_fit, ames_test) %>%
  bind_cols(ames_test %>% select(Sale_Price))

# Metrics
nn_metrics<-metric_set(rmse, mae, rsq)
print(nn_metrics(nn_preds, truth=Sale_Price, estimate=.pred))

# the current model is explaining 65.6% of the variance in the response
# log(sale price) might improve things


