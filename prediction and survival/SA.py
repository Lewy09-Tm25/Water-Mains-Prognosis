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
from lifelines import CoxPHFitter
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


# pre selected params for models
PARAMS_XGB_AFT = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': 'normal',
    'aft_loss_distribution_scale': 1.0,
    'tree_method': 'hist', 
    'learning_rate': 5e-2, 
    'max_depth': 8, 
    'booster':'dart',
    'subsample':0.5,
    'min_child_weight': 50,
    'colsample_bynode':0.5
}

PARAMS_XGB_COX = {
    'objective': 'survival:cox',
    'tree_method': 'hist', 
    'learning_rate': 5e-2, 
    'max_depth': 8, 
    'booster':'dart',
    'subsample':0.5,
    'min_child_weight': 50, 
    'colsample_bynode':0.5
}

PARAMS_TREE = {
    'objective': 'survival:cox',
    'eval_metric': 'cox-nloglik',
    'tree_method': 'hist', 
    'max_depth': 100, 
    'booster':'dart', 
    'subsample': 1.0,
    'min_child_weight': 50, 
    'colsample_bynode': 1.0
}

PARAMS_LR = {
    'C': 1e-3,
    'max_iter': 500
}

N_NEIGHBORS = 50


def read_datasets(path):
    return pd.read_csv(path)

def normalization(scaler, data, sc_cols):
    scaler.fit(data[sc_cols])
    data[sc_cols] = scaler.transform(data[sc_cols])
    return data


def cox_models(sa_data):
    # Cox-PHM
    surv_ress = []
    cscores = []
    penalizers = [0.0001,0.001,0.01,0.1,1,10]

    only_for_cox_data = sa_data.drop(['Installation_Year', 'Condition_score'], axis = 1)
    for pen in penalizers:
        wf = CoxPHFitter(penalizer=pen).fit(only_for_cox_data, "Break_Time", "Label")
        censored_subjects = only_for_cox_data.loc[~only_for_cox_data['Label'].astype(bool)]
        censored_subjects_last_obs = censored_subjects['Break_Time']

        # predict new survival function
        surv_ress.append(wf.predict_survival_function(censored_subjects, conditional_after=censored_subjects_last_obs))
        cscores.append(wf.concordance_index_)

    best_pen = penalizers[cscores.index(max(cscores))]

    only_for_cox_test = only_for_cox_data.sample(frac = 0.2)
    only_for_cox_train = only_for_cox_data.drop(only_for_cox_test.index)

    cph_best = CoxPHFitter(penalizer = best_pen)
    cph_best.fit(only_for_cox_train, duration_col='Break_Time', event_col='Label')
    censored_subjects = only_for_cox_test.loc[~only_for_cox_test['Label'].astype(bool)]
    censored_subjects_last_obs = censored_subjects['Break_Time']

    # predict new survival function
    cph_best.predict_survival_function(censored_subjects, conditional_after=censored_subjects_last_obs)
    joblib.dump(cph_best, "models/SA/Cox_SA_model.pkl")
    return cph_best.concordance_index_, cph_best


def data_split(sa_data):
    X = sa_data.drop(['Break_Time', 'Label', 'Installation_Year', 'Condition_score'], axis=1)
    y = convert_to_structured(sa_data['Break_Time'], sa_data['Label'])
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2)
    return X_train, X_valid, y_train, y_valid


def xgb_aft(X_train, X_valid, y_train, y_valid):
    # converting to xgboost format
    dtrain = convert_data_to_xgb_format(X_train, y_train, 'survival:aft')
    dval = convert_data_to_xgb_format(X_valid, y_valid, 'survival:aft')

    # training model
    bst = xgb.train(
        PARAMS_XGB_AFT,
        dtrain,
        verbose_eval=0
    )

    # predicting and evaluating
    preds = bst.predict(dval)
    cind = concordance_index(y_valid, -preds, risk_strategy='precomputed')
    print(f"C-index: {cind:.3f}")
    print(f"Average survival time: {preds.mean():.0f} years")
    joblib.dump(bst, "models/SA/XGB_AFT_model.pkl")
    return cind, bst


def xgb_km_n(X_train, X_valid, y_train, y_valid):
    # training model
    xgbse_kaplan = XGBSEKaplanNeighbors(PARAMS_XGB_AFT, n_neighbors=30)

    xgbse_kaplan.fit(
        X_train, y_train
    )

    # predicting
    preds = xgbse_kaplan.predict(X_valid)

    # running metrics
    print(f'C-index: {concordance_index(y_valid, preds)}')
    print(f'Avg. Brier Score: {approx_brier_score(y_valid, preds)}')
    print(f"""D-Calibration: {dist_calibration_score(y_valid, preds) > 0.05}""")
    joblib.dump(xgbse_kaplan, "models/SA/XGB_KM_N_model.pkl")
    return concordance_index(y_valid, preds), xgbse_kaplan


def xgb_tree(X_train, X_valid, y_train, y_valid):
    xgbse_km_tree = XGBSEKaplanTree(PARAMS_TREE)
    xgbse_km_tree.fit(
        X_train, y_train
    )
    # predicting
    preds = xgbse_km_tree.predict(X_valid)
    print(f'C-index: {concordance_index(y_valid, preds)}')
    print(f'Avg. Brier Score: {approx_brier_score(y_valid, preds)}')
    print(f"""D-Calibration: {dist_calibration_score(y_valid, preds) > 0.05}""")
    joblib.dump(xgbse_km_tree, "models/SA/XGB_KM_TREE_model.pkl")
    return concordance_index(y_valid, preds), xgbse_km_tree


def xgb_lor(X_train, X_valid, y_train, y_valid):
    base_model = XGBSEDebiasedBCE(PARAMS_XGB_AFT, PARAMS_LR)
    # fitting the meta estimator
    base_model.fit(
        X_train,
        y_train
    )
    # predicting
    preds = base_model.predict(X_valid)
    print(f'C-index: {concordance_index(y_valid, preds)}')
    print(f'Avg. Brier Score: {approx_brier_score(y_valid, preds)}')
    print(f"""D-Calibration: {dist_calibration_score(y_valid, preds) > 0.05}""")
    joblib.dump(base_model, "models/SA/XGB_LOR_model.pkl")
    return concordance_index(y_valid, preds), base_model
    


if __name__ == "__main__":

    # Read data
    dataset = read_datasets("data/Water_Main_Prognosis.csv")
    risk_pipes = read_datasets("data/risk_prone_pipes.csv")

    # Use columns
    sa_data_cols = ['PRESSURE_ZONE', 'PIPE_SIZE', 'Shape__Length', 'Installation_Year', 'Break_Time',
                'Label', 'Condition_score', 'MATERIAL']
    sa_data = dfss[sa_data_cols]
    sa_data = sa_data.reset_index(drop = True)

    sa_data.rename(columns = {'PIPE_SIZE':'PIPE_DIAMETER','Shape__Length':'PIPE_LENGTH'}, inplace = True)

    sa_data = pd.get_dummies(sa_data, columns = ['MATERIAL','PRESSURE_ZONE'])

    # Normalization of the data
    sc = StandardScaler()
    scale_cols = ['PIPE_LENGTH','PIPE_DIAMETER']
    sa_data = normalization(sc, sa_data, scale_cols)

    cindices = []
    cmodels = []

    # # Cox-PHM
    cicox, coxmodel = cox_models(sa_data)
    cindices.append(cicox)
    cmodels.append(coxmodel)

    # # splitting to X, T, E format
    X_train, X_valid, y_train, y_valid = data_split(sa_data)


    # # Fit XGBoost model to predict a value that is interpreted as a risk metric. Fit Weibull Regression model using risk metric as only independent variable.
    # # Predicts survival probabilities using the XGBoost + Weibull AFT stacking pipeline.
    xgbaftc, xgbaftmodel = xgb_aft(X_train, X_valid, y_train, y_valid)
    cindices.append(xgbaftc)
    cmodels.append(xgbaftmodel)


    # Transform feature space by fitting a XGBoost model and outputting its leaf indices. Build search index in the new space to allow nearest neighbor queries at scoring time.
    # Make queries to nearest neighbor search index build on the transformed XGBoost space. Compute a Kaplan-Meier estimator for each neighbor-set. Predict the KM estimators
    xgbkmnc, xgbkmnmodel = xgb_km_n(X_train, X_valid, y_train, y_valid)
    cindices.append(xgbkmnc)
    cmodels.append(xgbkmnmodel)


    # Fit a single decision tree using xgboost. For each leaf in the tree, build a Kaplan-Meier estimator.
    # Run samples through tree until terminal nodes. Predict the Kaplan-Meier estimator associated to the leaf node each sample ended into.
    xgbtc, xgbtmodel = xgb_tree(X_train, X_valid, y_train, y_valid)
    cindices.append(xgbtc)
    cmodels.append(xgbtmodel)


    # Predicts survival probabilities using the XGBoost + Logistic Regression pipeline.
    # Train a set of logistic regressions on top of the leaf embedding produced by XGBoost, each predicting survival at different user-defined discrete time windows. 
    # The classifiers remove individuals as they are censored, with targets that are indicators of surviving at each window.
    xgblorc, xgblormodel = xgb_lor(X_train, X_valid, y_train, y_valid)
    cindices.append(xgblorc)
    cmodels.append(xgblormodel)


    # This command will let us know which model performed the best. It is just read-only. With research done, it was concluded that XGB+LOR was the best model, with c-index of 0.81
    best_model = cmodels[cindices.index(max(cindices))]




    # Output a csv file with survival chances of the risk-prone pipes
    for_surv_pred_risk_pipes = risk_pipes.drop(['Age', 'Predicted condition', 'Label'], axis = 1)
    pred_model = joblib.load("models/SA/XGB_LOR_model.pkl")
    surv_cols = list(X_valid.columns)
    for_surv_pred_risk_pipes = for_surv_pred_risk_pipes[surv_cols]
    preds = pred_model.predict(for_surv_pred_risk_pipes)
    preds.to_csv("data/risk_prone_pipes_survival_chances.csv", index = True)

    # Now, pick any 4 pipes with proper indexing.
    # Storing indices first
    inds_for_surv_pred_risk_pipes = list(for_surv_pred_risk_pipes.index)
    # Randomly picking any 4 numbers from the range of 0 to len(for_surv_pred_risk_pipes)
    # NOTE:- Randomly picking 4 numbers gives the same trajectory for at least 2 pipes, tried multiple times. Hence, picking manually.
    rand_nums_inds = random.sample(range(0,len(for_surv_pred_risk_pipes)), 4)
    nums_inds = [181,179,3,1]
    final_df = risk_pipes.iloc[nums_inds]

    final_df_surv = preds.iloc[nums_inds]

    for i in range(len(final_df_surv)):
        plt.plot(final_df_surv.columns, final_df_surv.iloc[i], label=f'Pipe {i+1}')
    # Set labels and title
    plt.xlabel('Years')
    plt.ylabel('Survival Probability')
    # plt.title('Probability Values for Each Row')
    plt.grid(True)
    plt.legend()
    plt.savefig("graphs/surv_prob_4_pipes.png")

    # Have to see the graph and then determine. Has to be a better way. This is TODO
    # Based on final_df_surv, let's look at 55 year mark. blue decreses sharply, and hence the order should be 1,3,4,2.
    # so even if the 2nd pipe starts with low chance of survial, it is estimate to last longer than the other 3.
    order_of_repair = [1,4,2,3]
    final_df['Priority'] = order_of_repair
    data_to_send = final_df[['Predicted condition','Priority']]
    data_to_send.to_csv("data/DATA_FOR_CSP.csv", index = False)