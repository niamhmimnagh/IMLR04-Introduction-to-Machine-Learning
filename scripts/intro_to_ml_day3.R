# LOGISTIC REGRESSION FOR SPAM EMAILS

library(tidymodels)
library(DAAG)
library(dplyr)

# import spam dataset
data(spam7)
head(spam7)

str(spam7)

spam7<- spam7 %>%
  dplyr::mutate(yesno = factor(yesno, levels=c("n", "y"))) # ensure positive class is 'y'

# Train/test split (stratify by response to preserve the class balance)
library(rsample)
spam_split<-initial_split(spam7, prop=0.7, strata=yesno)

# proportion/balance of yes vs no in the original dataset
table(spam7$yesno)/nrow(spam7) # 0.605 no and 0.39 is yes

spam_train<-training(spam_split)
spam_test<-testing(spam_split)

# stratifying has maintained the original balance between yes and no
table(spam_train$yesno)/nrow(spam_train) # 0.605 no and 0.39 is yes
table(spam_test$yesno)/nrow(spam_test) # 0.605 no and 0.39 is yes

# BECAUSE WE ARE TUNING PENALTY/MIXTURE, WE NEED TO DIVIDE OUR TRAINING DATA
spam_folds<-vfold_cv(spam_train, v=5, strata=yesno)

# RECIPE
spam_rec<-recipe(yesno~., data=spam_train) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# MODEL SPECIFICATION
# regularised regression - separation
# separation - a combination of predictors perfectly predicts the response
# sounds like a good thing - bad for model stability
# error: 'fitted probabilities numerically 0 or 1 occurred'

# logit_glmnet_spec<-logistic_reg() %>% set_engine('glmnet) # without regularisation

# separation - perfectly separating the two classes
logit_glmnet_spec<-logistic_reg(
  penalty = tune(), # how much shrinkage occurs
  mixture = tune() # alpha - ridge, lasso or elastic net shrinkage. mixture=0 ridge, mixture=1 is lasso
) %>%
  set_engine('glmnet')


# Put our recipe and model spec together into a workflow
spam_glmnet_wf<-workflow()%>%
  add_model(logit_glmnet_spec)%>%
  add_recipe(spam_rec)


# TUNING STEPS
# specify our tuning grid
spam_grid<-grid_latin_hypercube(
  penalty(),  # by leaving this empty we are using the defaults
  mixture(),
  size=10
)
spam_grid

# Perform tuning with cross validation
# recall and sensitivity are the same thing
# we'll find out what it calls precision = positive predictive value
# try to use the terms your audience/stakeholders use

# cross validation results
spam_glmnet_res<-tune_grid(
  spam_glmnet_wf, #  workflow
  resamples = spam_folds, # where to test
  grid = spam_grid, # what values of the hyperparameters to check
  metrics=metric_set(roc_auc, accuracy, sens, ppv) # monitor these metrics
)

# Pick the best value:
best_glmnet<-select_best(spam_glmnet_res, metric='roc_auc')
# ROC AUC judges how well the model ranks positives above negatives across all possible cutoffs
# Trying to maximise the true positive rate (number of correctly identified positives) while 
# minimising the false positive rate (observations we incorrectly identify as positive)

# Finalise the workflow

spam_glmnet_final<-finalize_workflow(spam_glmnet_wf, best_glmnet) %>%
  fit(spam_train)


# Evaluate on test data
# we can extract the probabilities
# we can extract the predicted classes - threshold of 0.5
spam_glmnet_pred <- predict(spam_glmnet_final, spam_test, type =  'prob') %>%
  bind_cols(predict(spam_glmnet_final, spam_test, type = 'class')) %>%
  bind_cols(spam_test %>% dplyr::select(yesno))
head(spam_glmnet_pred) # we have our predicted probabilities, predicted classes and actual classes

levels(spam_glmnet_pred$.pred_class) # 'no' is first and 'yes' is second
# Extract Metrics

# Confusion matrix
conf_mat(data=spam_glmnet_pred,
         truth  = yesno,
         estimate = .pred_class)

# incorrectly identifying 40 'no' as 'yes'
# missing out on 206 of the 'yes' - classifying them as 'no'

# Accuracy
accuracy(data=spam_glmnet_pred,
         truth  = yesno,
         estimate = .pred_class)
# accuracy of 82.2% - doesn't reveal that the high-ish accuracy is coming from the well-predicted
# 'no'
# the overall percentage of correct predictions - 
# correctly predicted 82.2% of the observations

# Recall/ Sensitivity 
sens(data=spam_glmnet_pred,
     truth  = yesno,
     estimate = .pred_class,
     event_level = 'second')
# Recall =  0.62 or 62% 
# out of all the spam emails we are correctly identifying 62%
# we are missing 38% of the actual (true) spam
# if we want to catch more spam, we might need to decrease the threshold - try 0.4 or 0.3

# Precision
precision(data=spam_glmnet_pred,
          truth  = yesno,
          estimate = .pred_class,
          event_level = 'second')
# precision is 89.4%
# out of all the emails we predicted as spam - 89.4% were actually spam
# false positives - 10.6% of our predictions.

# do we want more genuine emails to be directed to spam, or do we want more spam to get through our
# filters

### MULTINOMIAL REGRESSION

# IRIS DATA

data(iris)
head(iris)

# Split data into training and testing
iris_split<-initial_split(iris, prop=0.7, strata=Species)
iris_train<-training(iris_split)
iris_test<-testing(iris_split)


# RECIPE
# real data is messy and probably requires more preproccessing
# Species~. means use all the other variables within the dataset to predict species
iris_rec<-recipe(Species~., data=iris_train) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())


# MODEL SPEC - multinomial logistic regression
multinom_spec<-multinom_reg()%>%
  set_engine("nnet") %>%
  set_mode("classification")


# WORKFLOW
iris_multinom_wf<-workflow()%>%
  add_model(multinom_spec) %>%
  add_recipe(iris_rec)


# FIT THE MODEL
iris_multinom_fit<-fit(iris_multinom_wf, data=iris_train)

# PREDICT ON THE TEST DATA
iris_multinom_pred<-predict(iris_multinom_fit, new_data=iris_test, type='prob') %>%
  bind_cols(predict(iris_multinom_fit, new_data=iris_test, type='class')) %>%
  bind_cols(iris_test %>% dplyr::select(Species))
head(iris_multinom_pred)  

# CONFUSION MATRIX
conf_mat(iris_multinom_pred, truth=Species, estimate = .pred_class)
# predicting setosa and virginica correctly
# very small confusion between versicolor and virginica


# RANDOM FOREST - for predicting spam email

# Train/Testing split already done - spam_train and spam_split

# We've also defined our recipe -  spam_rec

# We need to define our model specification

rf_spec_tunable<- rand_forest(
  mtry = tune(), # number of predictors to examine for each split
  trees = 1000,
  min_n = tune() # minimum number of observations that need to be inside a node for it to split
) %>%
  set_engine("ranger")%>%
  set_mode("classification")


# Create our workflow
spam_rf_wf<- workflow() %>%
  add_model(rf_spec_tunable) %>%
  add_recipe(spam_rec)


# Create a tuning grid
head(spam7)
# L means integer

spam_rf_grid<-grid_latin_hypercube(
  mtry(range=c(2L, 6L)),
  min_n(),
  size = 5
)
spam_rf_grid

# RUN THE CROSS VALIDATION TO CARRY OUT HYPERPARAMETER TUNING
spam_rf_res<-tune_grid(
  spam_rf_wf, 
  resamples = spam_folds, 
  grid= spam_rf_grid, 
)

# SELECT THE BEST HYPERPARAMETERS
spam_rf_best<-select_best(spam_rf_res, metric='roc_auc')

# finalise the workflow
spam_rf_final_wf<-finalize_workflow(spam_rf_wf, spam_rf_best)

# Fit on the full dataset
spam_rf_fit<-fit(spam_rf_final_wf, data=spam_train)


# Predict on the test data
spam_rf_pred<-predict(spam_rf_fit, new_data=spam_test, type='prob') %>%
  bind_cols(predict(spam_rf_fit, new_data=spam_test, type='class')) %>%
  bind_cols(spam_test %>% dplyr::select(yesno))


head(spam_rf_pred)


# Confusion Matrix
# lets us caret's confusion matrix
reference<-spam_rf_pred$yesno
levels(reference)

prediction<-spam_rf_pred$.pred_class
levels(prediction)

caret::confusionMatrix(data=prediction, reference=reference, positive='y')

# no information rate - accuracy you would get if you always predicted the majority (always predicting 'n')
# baseline to compare against

# Sensitivity  = recall
# specificity = TNR = TN/(TN+FP) = amount of non-spam that are correctly identified as not being spam
# the proportion of 'n' correctly identified as 'n'

# Positive predictive value - Precision
# when we predict spam ('y'), how often is it actually spam 

# Negative predictive value - 
# when we predict not spam ('n'), how often its actually not spam

# Prevalence - proportion of positive cases in the data

# Balanced accuracy = mean of the recall and the specificity : (TPR+TNR)/2
# more useful than accuracy when classes are imbalanced


