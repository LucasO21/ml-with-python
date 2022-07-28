# ML With Python

This repo contains notebooks of learning projects for implementing machine learning in python.

=========================================================================

### Project 1: Ames Housing Prediction 
This is my first data science project using python. The goal of this project was to explore some feature selection techniques as well as focus on implementation in python using scikit-learn, as well as the [feature engine](https://feature-engine.readthedocs.io/en/1.3.x/) python package.

* Dataset - For this project, I used the [ames dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from kaggle. The reason I chose this dataset was that it has a good combination of numeric and categorical features, 82 features in total. This gave me an opportunity to explore various feature selection, feature encoding and feature transformation methods.

* Approach - I first used [feature selection](https://github.com/LucasO21/ml-with-python/blob/main/ames-housing-prediction/ames_prediction_feature_selection.ipynb) methods to determine the most important for predicting *sale price*. My goal was to reduce the number of features to work with so as not to be overwhelmed. Using Recursive Feature Selection RFE, I settled on 38 features, 32 determined by RFE and 4 additional features necessary for feature engineering. I used One-Hot Encoding (OHE) for categorical variables and tested out several transformations for numeric features. Finally I tried out several models including regularized linear models (Ridge and Lasso), Random Forest and Xgboost.

* Results - The Xgboost model provided the best performance (in terms of RMSE). RMSE  on the test set after hyperparameter tuning was $27,253. A step in the right direction, given the average *sale price* in the dataset was $182,376 with a standard deviation of $81,168. However there is still alot that can improved in this project. 

* Next Steps - Rinse, recycle, repeat. Areas of improvement include better feature selection, better feature engineering (create alot more new features), better and more efficient data preprocessing (perhaps using custom transformers). I'm certain making these improvements will result in a much better model. 


### Project 2: Telecom Customer Churn with H2o AutoML
This project focused on ML with [H2o AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html). The goal of this project was to predict customer churn for a telecommunications company. 

* Dataset - Dataset for this project came from [Maven Analytics](https://www.mavenanalytics.io/blog/maven-churn-challenge). The dataset consisted of 37 featurs and 7043 observations.

* Approach - Performed EDA to understand how predictivive features relate to the target. Initial analysis showed that features like *tenure in months*, *contract status* might offer strong predictive value. After EDA, I proceeded to modeling with H2o. Additionally I used SHAP explanation to understand model predictions.

* Results - H2o AutoML produced several models that performed well on the train and test data. The AutoML Leader was a Stacked Ensemble. However I decided to investigage the best GBM model on the test data. The best GBM performed well on the test data with an AUC of 0.93, logloss of 0.27 and AUCPR of 0.97. Additionally, variable importance showed the top 5 important predictors to be *tenure in months*, *contract status*, *number of referrals*, *monthly charge* and *age* respectively.

* Netxt Steps - Develop deeper understanding of model explainability through variable importance and SHAP explanations. 

