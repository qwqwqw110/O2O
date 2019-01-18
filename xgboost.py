import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split

dataset1 = pd.read_csv('F:/Learning/competitions/TianChi/o2o/MyCode/dataFeature/ProcessDataSet1.csv')
dataset1.label.replace(-1, 0, inplace=True)
dataset2 = pd.read_csv('F:/Learning/competitions/TianChi/o2o/MyCode/dataFeature/ProcessDataSet2.csv')
dataset2.label.replace(-1, 0, inplace=True)
dataset3 = pd.read_csv('F:/Learning/competitions/TianChi/o2o/MyCode/dataFeature/ProcessDataSet3.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset12 = pd.concat([dataset1, dataset2], axis=0)
dataset12_y = dataset12.label
dataset12_x = dataset12.drop(['user_id', 'label', 'day_gap_before', 'coupon_id', 'day_gap_after'], axis=1)

dataset3.drop_duplicates(inplace=True)
dataset3_preds = dataset3[['user_id', 'coupon_id', 'date_received']]
dataset3_x = dataset3.drop(['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after'], axis=1)

dataTrain = xgb.DMatrix(dataset12_x, label=dataset12_y)
dataTest = xgb.DMatrix(dataset3_x)


# 性能评价函数
def myauc(test):
    testgroup = test.groupby(['coupon_id'])
    aucs = []
    for i in testgroup:
        tmpdf = i[1]
        if len(tmpdf['label'].unique()) != 2:
            continue
        fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred'], pos_label=1)
        aucs.append(auc(fpr, tpr))
    return np.average(aucs)


params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }
watchlist = [(dataTrain,'train')]
model = xgb.train(params,dataTrain,num_boost_round=3500,evals=watchlist)

model.save_model('F:/Learning/competitions/TianChi/o2o/MyCode/xgbmodel')
model=xgb.Booster(params)
model.load_model('F:/Learning/competitions/TianChi/o2o/MyCode/xgbmodel')
#predict test set
dataset3_preds1 = dataset3_preds
dataset3_preds1['label'] = model.predict(dataTest)
dataset3_preds1.label = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(dataset3_preds1.label.reshape(-1,1))
dataset3_preds1.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds1.to_csv("F:/Learning/competitions/TianChi/o2o/MyCode/xgb_preds.csv",index=None,header=None)
print(dataset3_preds1.describe())


model=xgb.Booster()
model.load_model('F:/Learning/competitions/TianChi/o2o/MyCode/xgbmodel')

temp = dataset12[['coupon_id','label']].copy()
temp['pred'] =model.predict(xgb.DMatrix(dataset12_x))
temp.pred = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(temp['pred'].values.reshape(-1,1))
print(myauc(temp))


params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }

cvresult = xgb.cv(params, dataTrain, num_boost_round=20000, nfold=5, metrics='auc', seed=0, callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(50)
        ])
num_round_best = cvresult.shape[0] - 1
print('Best round num: ', num_round_best)

watchlist = [(dataTrain,'train')]
model1 = xgb.train(params,dataTrain,num_boost_round=num_round_best,evals=watchlist)

model1.save_model('F:/Learning/competitions/TianChi/o2o/MyCode/xgbmodel1')
print('------------------------train done------------------------------')

model1=xgb.Booster()
model1.load_model('F:/Learning/competitions/TianChi/o2o/MyCode/xgbmodel1')

temp = dataset12[['coupon_id','label']].copy()
temp['pred'] =model1.predict(xgb.DMatrix(dataset12_x))
temp.pred = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(temp['pred'].values.reshape(-1,1))
print(myauc(temp))


dataset3_preds2 = dataset3_preds
dataset3_preds2['label'] = model1.predict(dataTest)
dataset3_preds2.label = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(dataset3_preds2.label.reshape(-1,1))
dataset3_preds2.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds2.to_csv("F:/Learning/competitions/TianChi/o2o/MyCode/xgb_preds2.csv",index=None,header=None)
print(dataset3_preds2.describe())

feature_score = model1.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)  # value逆序排序

fs = []
for (key, value) in feature_score:
    fs.append("{0},{1}\n".format(key, value))

with open('F:/Learning/competitions/TianChi/o2o/MyCode/xgb_feature_score.csv', 'w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)