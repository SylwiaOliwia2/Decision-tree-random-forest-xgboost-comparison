# Decision-tree-random-forest-xgboost-comparison
Comparison of decision trees , random forest and xgbbost in solving regressoin problem based on Kaggle 'House Prices' competition.
Link to my leaderboard: https://www.kaggle.com/sylwiamielnicka

### Take a note - this package require xgboost version 06-0.4-0 released on 12 May 2015. Version 06-0.4x released on 15th Jan 2017 has many changes in syntax.

The code was prepared for my undergraduate disertation, it's purpose wasn't best result on Kaggle, but comparing performance of decisin trees, random forest and xgboost (feature engineering is just a sample). Tree-based models were compared in three stages:
1. raw data (messy, with NA)
2. data with filled-in NA's (MICE implementation)
3. sample feature engineering + fine-tuning models' parameters 

Code is divided into three parts: 
**tree_rf_xgb_models.R** - contains main code
**funkcje.R** - contains custom functions which are used in tree_rf_xgb_models.R.
