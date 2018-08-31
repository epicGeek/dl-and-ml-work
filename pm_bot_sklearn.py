# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


if __name__ == '__main__':

    train = pd.read_csv('C:/Users/neilpei/Desktop/deep_wide/sale_data_pre-processed.csv')
    test = pd.read_csv('C:/Users/neilpei/Desktop/deep_wide/sale_data_test.csv')

    # 计算相关性
    corrmat = train.corr()
    k = 10
    cols = corrmat.nlargest(k, 'ship_qty')['ship_qty'].index
    cm = np.corrcoef(train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

    # 挑选特征值
    selected_features = ['sku_no', 'cust_no', 'week_number']

    X_train = train[selected_features]
    X_test = test[selected_features]
    y_train = train['ship_qty']

    # 进行特征向量化
    from sklearn.feature_extraction import DictVectorizer

    dict_vec = DictVectorizer(sparse=False)

    X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = dict_vec.transform(X_test.to_dict(orient='record'))

    # 用GradientBoostingRegressor进行预测
    from sklearn.ensemble import GradientBoostingRegressor

    rfr = GradientBoostingRegressor()
    rfr.fit(X_train, y_train)
    rfr_y_predict = rfr.predict(X_test)

    rfr_submission = pd.DataFrame({'ID': test['ID'], 'ship_qty': rfr_y_predict})
    rfr_submission.to_csv('C:/Users/neilpei/Desktop/deep_wide/infer.csv', index=False, sep=',')