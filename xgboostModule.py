import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split

#  读取预处理之后的数据
dataset1 = pd.read_csv('F:\Learning\competitions\TianChi\o2o\O2OProject\dataFeature\ProcessDataSet1.csv')
dataset1.label.replace(-1, 0, inplace=True)
dataset2 = pd.read_csv('F:\Learning\competitions\TianChi\o2o\O2OProject\dataFeature\ProcessDataSet2.csv')
dataset2.label.replace(-1, 0, inplace=True)
dataset3 = pd.read_csv('F:\Learning\competitions\TianChi\o2o\O2OProject\dataFeature\ProcessDataSet3.csv')
#删除重复数据
dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)

#  以行为轴，将dataset1，dataset2数据拼接起来
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

# 设置Xgboost的参数
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
# 进行训练，此次训练主要是为了和利用cv函数得到最优决策树数量之后的模型进行对比
watchlist = [(dataTrain,'train')]
model = xgb.train(params,dataTrain,num_boost_round=3500,evals=watchlist)
# 保存训练得到的模型
model.save_model('F:/Learning/competitions/TianChi/o2o/O2OProject/xgbmodel')
model=xgb.Booster(params)
model.load_model('F:/Learning/competitions/TianChi/o2o/O2OProject/xgbmodel')

# 对测试集进行预测
dataset3_preds1 = dataset3_preds
dataset3_preds1['label'] = model.predict(dataTest)
dataset3_preds1.label = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(dataset3_preds1.label.values.reshape(-1,1))  # n行1列
dataset3_preds1.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds1.to_csv("F:/Learning/competitions/TianChi/o2o/O2OProject/xgb_preds.csv",index=None,header=None)
print(dataset3_preds1.describe())


model=xgb.Booster()
model.load_model('F:/Learning/competitions/TianChi/o2o/O2OProject/xgbmodel')
# 对训练集进行预测，平均得到AUC
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

#  得到最优决策树数量
cvresult = xgb.cv(params, dataTrain, num_boost_round=20000, nfold=5, metrics='auc', seed=0, callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(50)
        ])

num_round_best = cvresult.shape[0] - 1
print('Best round num: ', num_round_best)
# 利用得到的最优决策树数量再次对模型进行训练
watchlist = [(dataTrain,'train')]
model1 = xgb.train(params,dataTrain,num_boost_round=num_round_best,evals=watchlist)
# 保存再次训练的模型
model1.save_model('F:/Learning/competitions/TianChi/o2o/O2OProject/xgbmodel1')
print('------------------------train done------------------------------')

model1=xgb.Booster()
model1.load_model('F:/Learning/competitions/TianChi/o2o/O2OProject/xgbmodel1')
# 利用训练集对新模型进行测试得到平均AUC
temp = dataset12[['coupon_id','label']].copy()
temp['pred'] =model1.predict(xgb.DMatrix(dataset12_x))
temp.pred = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(temp['pred'].values.reshape(-1,1))
print(myauc(temp))

# 利用新模型对测试集进行预测
dataset3_preds2 = dataset3_preds
dataset3_preds2['label'] = model1.predict(dataTest)
dataset3_preds2.label = MinMaxScaler(copy=True,feature_range=(0,1)).fit_transform(dataset3_preds2.label.values.reshape(-1,1))
dataset3_preds2.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds2.to_csv("F:/Learning/competitions/TianChi/o2o/O2OProject/xgb_preds2.csv",index=None,header=None)
print(dataset3_preds2.describe())
# 对每个特征进行评分
feature_score = model1.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)  # value逆序排序

fs = []
for (key, value) in feature_score:
    fs.append("{0},{1}\n".format(key, value))
# 保存特征评分
with open('F:/Learning/competitions/TianChi/o2o/O2OProject/xgb_feature_score.csv', 'w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)
