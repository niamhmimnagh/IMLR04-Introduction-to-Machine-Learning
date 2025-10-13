library(skimr)
library(DataExplorer)
library(tidyverse)
library(titanic)
library(stringr)
library(dplyr)

# assign iris dataset to 'dat'
dat<-as_tibble(iris)

# look at the first 6 rows
dat %>% slice_head(n=6)

# dimension - 150 rows and 5 columns
dim(dat)

# names of the columns
names(dat)

# data types - in base R
str(dat)

# skimr::skim
skimr::skim(dat)

# Class balance for the target
dat %>% 
  count(Species, name='n') %>%
  mutate(prop=n/sum(n))

# 3 rows - one for each species - 50 observations for each of our categories
# proportion - each of our categories is 1/3 of the dataset
# this is nice and balanced for classification models since no class dominates the target variable

# Quick missingness check

colSums(is.na(dat))
  
  
# DataExplorer Plots
# intro plot - tells you the rows and columns, continuous vs. discrete, missing percentage
# good for quick confirmation that all is as it should be
plot_intro(dat)

# missingness by predictor - 0 for all, but good for real data
plot_missing(dat)

# categorical distributions
# lets you know if your categories are balanced nicely or not
plot_bar(dat)

# numeric variables
# automatically detects numeric variables
# helps reveal skewness or multimodality
# petal length and petal width show 2 clear peaks - bimodal distribution
# suggests strong difference - possibly subgroups present in the 
# no extreme outliers visible, and ranges consistent with expected biological variation
plot_histogram(dat, ncol=2)
  
plot_density(dat, ncol=2)

# Recipes example with titanic dataset
library(titanic)
data("titanic_train")
head(titanic_train)


str(titanic_train)

dat<-titanic_train %>%
  as_tibble() %>%
  mutate(
    Survived =  factor(Survived, levels=c(0,1), labels=c("no", "yes")),
    Pclass = factor(Pclass)
  )

head(dat)
str(dat)

# have a quick look with skimr

skimr::skim(dat)

# sibsp = number of siblings or spouses
# parch =  number of parents/children aboard for this passenger

# empty strings for embarked and cabin - recode as NA
# rare categories might need handling -  
# step_other  - takes rare categories and bundles them together into an 'other' category
# step_unknown -  create an 'unknown' column
# remove variables that don't tell us anything - passengerID, name, ticket
# feature engineering - family size = SibSp +  Parch + 1
# IsAlone = if family size = 1 
# interaction = sex x class
# scale and encode any predictors - 
# for numeric variables we'll scale them, and for categorical variables we will encode them 0/1

# IMPORTANT - split before preprocessing
library(rsample)
# proportion = 0.75 - 75% of observations in the dataset
# 5/8 are 'no' and 3/8 are 'yes' - this will be roughly the same in the training dataset
split<-initial_split(dat, prop=0.75, strata=Survived)
train<-training(split)
test<-testing(split)

# we want to create a recipe - 
rec<-recipe(Survived~., data=train)

# remove any non-predictive/high-missing variables
rec1<-rec %>%
  step_rm(PassengerId, Name, Ticket, Cabin) %>% # drop/remove these variables
  step_zv(all_predictors()) # remove 'zero-variance' variables 
# all_predictors - tells the recipe to go through all variables and check if they have 0 variance or not

# impute any missing values
# median for numeric variables
# mode for categorical variables

rec2<-rec1%>%
  step_impute_median(all_numeric_predictors())%>%
  step_impute_mode(all_nominal_predictors())

# handle any rare or unknown categories
# if we have rare categories (less than 5% of the dataset) - bundle them together into an 'other' column
# make any unknown (NA) explicit

rec3<-rec2 %>%
  step_other(all_nominal_predictors(), threshold = 0.05)%>%
  step_unknown(all_nominal_predictors())




