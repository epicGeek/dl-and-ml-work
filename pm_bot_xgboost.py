import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import sqrt


def __get_feature(__df_x_train):
    data_num = len(__df_x_train)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        # tmp_list.append(float(__df_x_train.iloc[row]['oplgm_amt']))
        tmp_list.append(int(__df_x_train.iloc[row]['sku_no']))
        tmp_list.append(int(__df_x_train.iloc[row]['week_number']))
        XList.append(tmp_list)
    yList = __df_x_train.ship_qty.values
    idList = __df_x_train.id.values
    return XList, yList, idList


def __load_test_data(__df_x_test):
    data_num = len(__df_x_test)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        # tmp_list.append(float(__df_x_test.iloc[row]['oplgm_amt']))
        tmp_list.append(int(__df_x_test.iloc[row]['sku_no']))
        tmp_list.append(int(__df_x_test.iloc[row]['week_number']))
        XList.append(tmp_list)
    idList = __df_x_test.id.values
    return XList, idList


def __train_data(__x_train, __y_train, __x_test, df_x_test, test_id_list):
    # model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=100, objective='reg:gamma', silent=0, min_child_weight=1)
    model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100, objective='reg:gamma', silent=0, min_child_weight=1)
    model.fit(__x_train, __y_train)
    ans = model.predict(__x_test)
    ans_len = len(ans)
    # id_list = np.arange(len(__x_train)+1, len(__x_train)+len(__x_test)+1)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(test_id_list[row]), ans[row]])
    np_data = np.array(data_arr)
    pd_data = pd.DataFrame(np_data, columns=['id', 'ship_qty'])
    pd_data['id'] = pd_data['id'].astype('int')
    # print(pd_data)
    # plot_importance(model)
    # plt.show()
    pd_data = pd_data.merge(df_x_test, on='id', how='inner')
    pd_data = pd_data.sort_values(by='id')
    pd_data.to_csv('C:/Users/neilpei/Desktop/deep_wide/sale_data_infer.csv', index=None)
    return pd_data


def __model_evaluation(df_x_validation, predict_data):
    df_x_validation = df_x_validation.reset_index()
    predict_data = predict_data.reset_index()
    del predict_data['index']
    rmspe_add = 0
    mae_add = 0
    for index in range(0, len(df_x_validation)):
        real = df_x_validation.loc[index, 'ship_qty']
        predict = predict_data.loc[index, 'ship_qty']
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
    df_x_train.to_csv('C:/Users/neilpei/Desktop/deep_wide/sale_data_train_set.csv', index=False)
    df_x_test.to_csv('C:/Users/neilpei/Desktop/deep_wide/sale_data_test_set.csv', index=False)
    df_x_validation = df_x_test.copy(deep=True).sort_values(by='id')
    del df_x_test['ship_qty']
    x_train, y_train, train_id_List = __get_feature(df_x_train)
    x_test, test_id_list = __load_test_data(df_x_test)
    predict_data = __train_data(x_train, y_train, x_test, df_x_test, test_id_list)
    P_dict = __model_evaluation(df_x_validation, predict_data)

    id_array = df_x_validation['id'].values
    real_array = df_x_validation['ship_qty'].values
    predict_array = predict_data['ship_qty'].values

    plt.plot(id_array, real_array, label='real ship qty', linewidth=1, color='b', marker='o', markerfacecolor='blue',
             markersize=6)
    plt.plot(id_array, predict_array, label='predict ship qty', color='r')
    plt.xlabel('id.')
    plt.ylabel('Ship Qty')
    plt.title('Ship qty predict')
    plt.legend()
    plt.show()

    print P_dict