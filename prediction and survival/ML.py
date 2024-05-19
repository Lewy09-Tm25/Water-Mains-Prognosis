import sys
import os
import joblib
import pandas as pd
import numpy as np
from numpy import loadtxt
import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from xgbse.metrics import concordance_index,approx_brier_score,dist_calibration_score
from xgbse.non_parametric import get_time_bins
from xgbse import (
    XGBSEKaplanNeighbors,
    XGBSEKaplanTree,
    XGBSEDebiasedBCE,
    XGBSEBootstrapEstimator
)
from xgbse.converters import (
    convert_data_to_xgb_format,
    convert_to_structured
)



def read_datasets(path):
    return pd.read_csv(path)


def get_age(data,feature):
    data['Age'] = data['Installation_Year'].apply(lambda x: 2024 - x)
    data = data.drop(['Installation_Year'], axis = 1)
    data = data.loc[data['Age'] > 0]
    return data


def normalization(scaler, data, sc_cols):
    scaler.fit(data[sc_cols])
    data[sc_cols] = scaler.transform(data[sc_cols])
    return data


def modeling(data):
    test = data.sample(frac = 0.2)
    train = data.drop(test.index)
    x = train.drop(['Condition_score', 'Label', 'Break_Time'], axis = 1)
    y = train['Condition_score']
    rf = RandomForestRegressor
    knns = KNeighborsRegressor
    xgbs = xgb.XGBRegressor
    lgbs = lgb.LGBMRegressor
    xtest = test.drop(['Condition_score', 'Label', 'Break_Time'], axis = 1)
    ytest = test['Condition_score']

    models = [rf,knns,xgbs,lgbs]
    evals = {}
    for model in models:
        m = model()
        m.fit(x,y)
        preds = m.predict(xtest)
        evals[model.__name__] = [mean_absolute_error(ytest,preds),mean_squared_error(ytest,preds), r2_score(ytest,preds)]

    models = list(evals.keys())
    mae_scores = [evals[model][0] for model in models]
    mse_scores = [evals[model][1] for model in models]
    r2_scores = [evals[model][2] for model in models]

    return x, y, xtest, ytest, models, mae_scores, mse_scores, r2_scores


def model_graphs_plot(ms, maes, mses, r2s):
    x = np.arange(len(ms))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting MAE scores
    bar1 = ax1.bar(x - width/2, maes, width, label='MAE', color='green')

    # Instantiate a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()  

    # Plotting MSE scores
    bar2 = ax2.bar(x + width/2, mses, width, label='MSE', color='blue')

    # Adding some text for labels, title and custom x-axis tick labels, etc.
    ax1.set_xlabel('Models')
    ax1.set_ylabel('MAE', color='green')
    ax2.set_ylabel('MSE', color='blue')
    ax1.set_title('Model Performance: MAE, MSE, and R2 Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ms)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    fig.tight_layout()  # to ensure the right y-label is not slightly clipped
    plt.close(fig)
    plt.savefig("graphs/model_graphs.png")


def feature_importance_plot(data):
    x = data.drop(['Condition_score', 'Label', 'Break_Time'], axis = 1)
    y = data['Condition_score']
    fea_model = xgb.XGBRegressor()
    fea_model.fit(x, y)
    # Get feature importance scores and feature names
    feature_importance = fea_model.get_booster().get_score(importance_type='weight')
    keys = list(feature_importance.keys())
    values = list(feature_importance.values())

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame(data={'Feature': keys, 'Importance': values})

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(3)

    # Plot
    fig = plt.figure(figsize=(8, 4))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.close(fig)
    plt.savefig("graphs/feat_imp.png")



def prediction_filtering(x, y, xtest, ytest, model):
    model.fit(x,y)
    preds = model.predict(xtest)
    joblib.dump(model,"models/ML/ML_Model.pkl")

    predlist = list(preds)
    # Combine predictions with the 'Break' column from the test set
    result_df = xtest.copy()
    result_df['Predicted condition'] = predlist
    result_df['Label'] = test.loc[xtest.index, 'Label']

    # According to our dataset, the mean of of the condition score column is 8.5. Hence, we will filter out those pipes that are prone.
    risk_pipes = result_df.loc[result_df['Predicted condition'] < 8.5]
    return risk_pipes


if __name__ == "__main__":

    # Read data
    dataset = read_datasets("data/Water_Main_Prognosis.csv")

    # Use columns
    ml_data_cols = ['PRESSURE_ZONE', 'PIPE_SIZE', 'Shape__Length', 'Installation_Year', 'Break_Time',
                'Label', 'Condition_score', 'MATERIAL']
    ml_data = dfss[ml_data_cols]
    ml_data = ml_data.reset_index(drop = True)

    # Get 'Age' from 'Installation_Year'
    ml_data = get_age(ml_data, 'Installation_Year')
    ml_data.rename(columns = {'PIPE_SIZE':'PIPE_DIAMETER','Shape__Length':'PIPE_LENGTH'}, inplace = True)

    # Handling categorical values
    ml_data = pd.get_dummies(ml_data, columns=['PRESSURE_ZONE','MATERIAL'])

    # Normalization of the data
    sc = StandardScaler()
    scale_cols = ['PIPE_LENGTH','PIPE_DIAMETER','Age']
    ml_data = normalization(sc, ml_data, scale_cols)

    # Modeling with split sets, scores, and models as the returning elements
    x, y, xtest, ytest, models, mae_scores, mse_scores, r2_scores = modeling(ml_data)

    # Plot model performances
    model_graphs_plot(models, mae_scores, mse_scores, r2_scores)

    # Plot feature importance
    feature_importance_plot(ml_data)

    # The best performing model was Random Forest. So, will use that to predict the condition. Filtered pipes are in stored in a csv file that will be used by Survival Analysis
    rfmodel = RandomForestRegressor()
    filter_data = prediction_filtering(rfmodel)
    filter_data.to_csv("data/risk_prone_pipes.csv", index = True)