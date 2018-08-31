import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from math import sqrt


def __model_evaluation(df_x_validation):
    df_x_validation = df_x_validation.reset_index()
    rmspe_add = 0
    mae_add = 0
    for index in range(0, len(df_x_validation)):
        real = df_x_validation.loc[index, 'ship_qty_x']
        predict = df_x_validation.loc[index, 'ship_qty_y']
        rmspe_add += pow((real - predict), 2)
        mae_add += abs(real-predict)
    # MAE(Mean Absolute Error)
    MSE = rmspe_add / len(df_x_validation)
    RMSE = sqrt(MSE)
    # MSE(Mean Square Error)
    MAE = mae_add / len(df_x_validation)
    evaluation_dict = {
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE
    }
    return evaluation_dict



if __name__ == '__main__':
    data = pd.read_csv('C:/Users/neilpei/Desktop/deep_wide/sale_data_filled.csv')
    df_x = data.copy(deep=True)
    df_y = data.copy(deep=True)
    del df_y['oplgm_amt']
    del df_y['sku_no']
    del df_y['week_number']
    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0, shuffle=True)
    df_x_train.to_csv('C:/Users/neilpei/Desktop/deep_wide/sale_data_train_set_gbdt.csv', index=False)
    df_x_test.to_csv('C:/Users/neilpei/Desktop/deep_wide/sale_data_test_set_gbdt.csv', index=False)

    selected_features = ['sku_no', 'week_number']
    X_train = df_x_train[selected_features]
    X_test = df_x_test[selected_features]
    y_train = df_x_train['ship_qty']

    from sklearn.feature_extraction import DictVectorizer
    dict_vec = DictVectorizer(sparse=False)
    X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = dict_vec.transform(X_test.to_dict(orient='record'))

    from sklearn.ensemble import GradientBoostingRegressor

    rfr = GradientBoostingRegressor()
    rfr.fit(X_train, y_train)
    rfr_y_predict = rfr.predict(X_test)

    rfr_submission = pd.DataFrame({'id': df_x_test['id'], 'ship_qty': rfr_y_predict})
    rfr_submission = rfr_submission.sort_values(by='id')
    df_x_test = df_x_test.sort_values(by='id')
    df_gbdt_infer = df_x_test.merge(rfr_submission, on='id', how='inner')
    df_gbdt_infer.to_csv('C:/Users/neilpei/Desktop/deep_wide/sale_data_infer_gbdt.csv', index=False, sep=',')

    id_array = df_gbdt_infer['id'].values
    real_array = df_gbdt_infer['ship_qty_x'].values
    predict_array = df_gbdt_infer['ship_qty_y'].values

    plt.plot(id_array, real_array, label='real ship qty', linewidth=1, color='b', marker='o', markerfacecolor='blue',
             markersize=6)
    plt.plot(id_array, predict_array, label='predict ship qty', color='r')
    plt.xlabel('id.')
    plt.ylabel('Ship Qty')
    plt.title('Ship qty predict by GBDT')
    plt.legend()
    plt.show()
    P_dict = __model_evaluation(df_gbdt_infer)
    print P_dict