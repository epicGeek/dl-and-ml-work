# Import libs

import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from math import sqrt


# Constants & configuration
ROOT_DIR = 'C:/Users/neilpei/Desktop/deep_wide/by_month/ship_qty/'
MONTH_DATA_PATH = ROOT_DIR + 'dw_orders_month.csv'
ALL_DATA_PATH = ROOT_DIR + 'dw_orders_all.csv'
OUT_PUT_FILLED_DATA = ROOT_DIR + 'filled_by_month.csv'
INFER_DATA_OUT_PUT = ROOT_DIR + 'infer_by_month.csv'
TRAIN_SET_OUTPUT = ROOT_DIR + 'train_by_month.csv'
TEST_SET_OUTPUT = ROOT_DIR + 'test_by_month.csv'
FUTURE_X_TEST = ROOT_DIR + 'future_x_test_by_month.csv'
NEXT_YEAR_INFER = ROOT_DIR + 'next_year_infer_by_month.csv'
SKU_REMOVE_LIST = ['3770593', '4132640', '4132668', '3621019']
MONTH_CODE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
SEASON_CODE = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1]
FEATURE_VALUE = [ 'group_id', 'month', 'prod_code', 'year', 'season']


def __fill_data(__df_pm_by_month):
    print "origin length: %d" % len(__df_pm_by_month)
    __df_pm_by_month['filled'] = 0
    try:
        del __df_pm_by_month['id']
    finally:
        pass
    __df_new_data = pd.DataFrame()
    for index, row in __df_pm_by_month.iterrows():
        if (index + 1) == len(__df_pm_by_month):
            print "Already last line."
            break
        sku_no = int(row['sku_no'])
        year = int(row['year'])
        month = int(row['month'])
        prod_code = int(row['prod_code'])
        group_id = int(row['group_id'])
        next_line_sku = int(__df_pm_by_month.loc[index + 1, 'sku_no'])
        next_line_year = int(__df_pm_by_month.loc[index + 1, 'year'])
        next_line_month = int(__df_pm_by_month.loc[index + 1, 'month'])
        next_line_prod_code = int(__df_pm_by_month.loc[index + 1, 'prod_code'])
        next_line_group_id = int(__df_pm_by_month.loc[index + 1, 'group_id'])
        change_year_fill_data = (sku_no == next_line_sku
                                 and prod_code == next_line_prod_code
                                 and group_id == next_line_group_id
                                 and (year + 1) == next_line_year
                                 and month != 12)
        fill_data_in_same_year = (sku_no == next_line_sku
                                  and year == next_line_year
                                  and month != 12
                                  and (month+1) != next_line_month
                                  and prod_code == next_line_prod_code
                                  and group_id == next_line_group_id)
        if fill_data_in_same_year or change_year_fill_data:
            if fill_data_in_same_year:
                print "Month is not continuous! sku: %d, year: %d, month: %d" % (sku_no, year, month)
                delta_month = next_line_month - month
            if change_year_fill_data:
                print "Year changed! sku: %d, year: %d, month: %d" % (sku_no, year, month)
                delta_month = 12 - month + 1
            index = 1
            while index < delta_month:
                row = pd.Series({'ship_qty': 0, 'sku_no': sku_no,  'year': year,
                                 'month': month + index, 'prod_code': prod_code,
                                 'group_id': group_id, 'money': 0.0, 'filled': 1})
                __df_new_data = __df_new_data.append(row, ignore_index=True)
                index += 1

    __df_new_data['sku_no'] = __df_new_data['sku_no'].astype('int')
    __df_new_data['ship_qty'] = __df_new_data['ship_qty'].astype('int')
    __df_new_data['year'] = __df_new_data['year'].astype('int')
    __df_new_data['month'] = __df_new_data['month'].astype('int')
    __df_new_data['prod_code'] = __df_new_data['prod_code'].astype('int')
    __df_new_data['group_id'] = __df_new_data['group_id'].astype('int')
    __df_new_data['money'] = __df_new_data['money'].astype('float')
    __df_new_data['filled'] = __df_new_data['filled'].astype('int')
    __df_pm_by_month = __df_pm_by_month.append(__df_new_data, ignore_index=True)
    __df_pm_by_month = __df_pm_by_month.sort_values(by=['sku_no', 'prod_code', 'group_id', 'year', 'month'])
    print "After data fill, length: %d" % len(__df_pm_by_month)
    __df_pm_by_month = __df_pm_by_month.reset_index()
    del __df_pm_by_month['index']
    return __df_pm_by_month


def __data_pre_process(__df_month_data_filled):
    # kick out these sku data
    __df_month_data_filled = __df_month_data_filled[~__df_month_data_filled['sku_no'].isin(SKU_REMOVE_LIST)]
    __df_money_mean = __df_month_data_filled.groupby(by=['sku_no', 'year', 'group_id', 'prod_code']).mean()
    __df_money_mean.rename(columns={'money': 'year_avg'}, inplace=True)
    __df_money_mean = __df_money_mean.reset_index()
    del __df_money_mean['filled']
    del __df_money_mean['month']
    del __df_money_mean['ship_qty']
    __df_month_data_filled = __df_month_data_filled.merge(__df_money_mean, on=['sku_no', 'year', 'group_id', 'prod_code'])
    __df_month_data_filled['sale_ratio'] = __df_month_data_filled['money'] / __df_month_data_filled['year_avg']

    __df_season = pd.DataFrame({'month': MONTH_CODE, 'season': SEASON_CODE})
    __df_month_data_filled = __df_month_data_filled.merge(__df_season, on='month')
    return __df_month_data_filled


def __show_heat_map(__df_month_handled):
    corrmat = __df_month_handled.corr()
    k = 10
    cols = corrmat.nlargest(k, 'money')['money'].index
    cm = np.corrcoef(__df_month_handled[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True,
                     square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values)
    plt.show()


def __model_evaluation(predict_data, label_name):
    predict_data = predict_data.reset_index()
    rmspe_add = 0
    mae_add = 0
    for index in range(0, len(predict_data)):
        real = predict_data.loc[index, label_name]
        predict = predict_data.loc[index, label_name + "_infer"]
        rmspe_add += pow((real - predict), 2)
        mae_add += abs(real-predict)
    # MAE(Mean Absolute Error)
    MSE = rmspe_add / len(predict_data)
    RMSE = sqrt(MSE)
    # MSE(Mean Square Error)
    MAE = mae_add / len(predict_data)
    evaluation_dict = {
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE
    }
    return evaluation_dict


def __get_feature(__df_x_train):
    data_num = len(__df_x_train)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        for feature in FEATURE_VALUE:
            tmp_list.append(int(__df_x_train.iloc[row][feature]))
        XList.append(tmp_list)
    yList = __df_x_train.ship_qty.values
    idList = __df_x_train.id.values
    return XList, yList, idList


def __load_test_data(__df_x_test):
    data_num = len(__df_x_test)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        for feature in FEATURE_VALUE:
            tmp_list.append(int(__df_x_test.iloc[row][feature]))
        XList.append(tmp_list)
    idList = __df_x_test.id.values
    return XList, idList


# train model
def __train_data(__x_train, __y_train, __x_test, df_x_test, test_id_list):
    cv_params = {'gamma': [0,1]}
    other_params = {'learning_rate': 0.03, 'n_estimators': 280, 'max_depth': 7, 'min_child_weight': 30, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 1, 'gamma': 1, 'reg_alpha': 0, 'reg_lambda': 1}
    model = xgb.XGBRegressor(**other_params)
    model.fit(__x_train, __y_train,  verbose=True, eval_metric='rmse')
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(__x_train, __y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('every iter:{0}'.format(evalute_result))
    print('best param:{0}'.format(optimized_GBM.best_params_))
    print('best score:{0}'.format(optimized_GBM.best_score_))
    ans = model.predict(__x_test)
    ans_len = len(ans)
    # id_list = np.arange(len(__x_train)+1, len(__x_train)+len(__x_test)+1)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(test_id_list[row]), ans[row]])
    np_data = np.array(data_arr)
    pd_data = pd.DataFrame(np_data, columns=['id', 'ship_qty_infer'])
    pd_data['id'] = pd_data['id'].astype('int')
    # print(pd_data)
    # plot_importance(model)
    # plt.show()
    pd_data = pd_data.merge(df_x_test, on='id', how='inner')
    pd_data = pd_data.sort_values(by='id')
    pd_data.to_csv(INFER_DATA_OUT_PUT, index=None)
    return pd_data, model


def __split_data_set(__output_filled_data_path):
    __df_handled = pd.read_csv(__output_filled_data_path)
    df_x = __df_handled.copy(deep=True)
    df_y = __df_handled.copy(deep=True)
    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0, shuffle=True)
    df_x_train = df_x_train.sort_values(by='id')
    df_x_test = df_x_test.sort_values(by='id')
    df_y_train = df_y_train.sort_values(by='id')
    df_y_test = df_y_test.sort_values(by='id')
    df_x_train.to_csv(TRAIN_SET_OUTPUT, index=False)
    df_x_test.to_csv(TEST_SET_OUTPUT, index=False)
    return df_x_train, df_x_test, df_y_train, df_y_test


def __show_infer_plot(__predict_data):
    id_array = __predict_data['id'].values
    real_array = __predict_data['ship_qty'].values
    predict_array = __predict_data['ship_qty_infer'].values

    plt.plot(id_array, real_array, label='real sales', linewidth=1, color='b', marker='o', markerfacecolor='blue',
             markersize=6)
    plt.plot(id_array, predict_array, label='predict sales', color='r')
    plt.xlabel('id.')
    plt.ylabel('sales amt')
    plt.title('sales predict')
    plt.legend()
    plt.show()


def __predict_future(__df_month_handled):
    __df_future = __df_month_handled.copy(deep=True)
    del __df_future['filled']
    del __df_future['money']
    del __df_future['month']
    del __df_future['ship_qty']
    del __df_future['year']
    del __df_future['year_avg']
    del __df_future['sale_ratio']
    del __df_future['season']
    __df_future = __df_future.drop_duplicates(keep='first')
    __df_future = __df_future.reset_index()
    __df_future['year'] = 2018
    del __df_future['index']
    _year = [2018] * 12
    __df_date = pd.DataFrame({'year': _year, 'month': MONTH_CODE, 'season': SEASON_CODE})
    __df_future = __df_future.merge(__df_date, on='year')
    __df_future_next_year = __df_future.copy(deep=True)
    __df_future_next_year['year'] = 2019
    __df_future = __df_future.append(__df_future_next_year)
    __df_future.to_csv(FUTURE_X_TEST, index_label='id')
    __df_future = pd.read_csv(FUTURE_X_TEST)
    return __df_future


def __show_season_infer_plot(__x_axis, __y_axis):
    plt.xticks(np.arange(len(__x_axis)), __x_axis)
    plt.plot( __y_axis)
    # plt.xlabel('Date')
    # plt.ylabel('sales amt')
    # plt.title('sales predict')
    # plt.legend()
    plt.show()

def __get_price_info(ALL_DATA_PATH):
    df_price =
if __name__ == '__main__':
    # Fill data
    df_month_data = pd.read_csv(MONTH_DATA_PATH)
    # Get price data from all data set
    df_price = __get_price_info(ALL_DATA_PATH)
    df_month_data_filled = __fill_data(df_month_data)
    # Data pre-process
    df_month_handled = __data_pre_process(df_month_data_filled)
    # Out put filled data
    df_month_handled.to_csv(OUT_PUT_FILLED_DATA, index_label='id')
    # Heat map
    # __show_heat_map(df_month_handled)
    # Split data set
    df_x_train, df_x_test, df_y_train, df_y_test = __split_data_set(OUT_PUT_FILLED_DATA)
    # get feature
    x_train, y_train, train_id_List = __get_feature(df_x_train)
    x_test, test_id_list = __load_test_data(df_x_test)
    # Prediction evaluation
    predict_data, model = __train_data(x_train, y_train, x_test, df_x_test, test_id_list)
    # Calculate RMSE
    p_dict = __model_evaluation(predict_data, 'ship_qty')
    # Show evaluation and plot
    __show_infer_plot(predict_data)
    print p_dict
    # Generate test data: 2018-01 ~ 2019-12
    df_future_x_test = __predict_future(df_month_handled)
    x_future_test_list, x_future_test_id_list = __load_test_data(df_future_x_test)
    # predict next year data
    future_predict = model.predict(x_future_test_list)
    # Save infer data
    df_future_x_test['ship_qty_infer'] = future_predict
    df_future_x_test.to_csv(NEXT_YEAR_INFER, index=False)
    df_future_sum = df_future_x_test.groupby(['year', 'month']).sum()
    x_axis = df_future_sum.index.values
    y_axis = df_future_sum['ship_qty_infer'].values
    x_axis_list = []
    for t in x_axis:
        x_axis_list.append(str(t[0]) + '-' + str(t[1]))
    # Show plot
    __show_season_infer_plot(x_axis_list, y_axis)
    print len(future_predict)
