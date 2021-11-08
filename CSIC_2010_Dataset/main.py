# import pandas as pd
# import re
# import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import plot_importance
# import matplotlib.pyplot as plt
# %matplotlib inline

def parsing(path):
    with open(path, 'r', encoding='utf-8') as f: # 파일 열기
        data = []
        para = ""
        
        line = "parsing start"
        while line:
            line = f.readline() # 한 줄씩 read
                
            if line.startswith("GET"): # GET 부분
                para += line
                data.append(para)
                para = ""
            
            # POST and PUT 부분
            elif line.startswith("POST") or line.startswith("PUT"):
                para += (line + " ")
                l = ""
                while not l.startswith("Content-Length"):
                    l = f.readline()
                l = f.readline()
                l = f.readline()
                para += l
                data.append(para)
                para = ""
                    
    return data


def dataset(path, mod='train'):
    x = parsing(f'{path}norm_{mod}.txt') # 데이터셋을 생성합니다. 파싱한 데이터와 라벨을 생성합니다 
    y = [0] * len(x) # 데이터셋을 생성합니다. 파싱한 데이터와 라벨을 생성합니다 
    
    x += parsing(f'{path}anomal_{mod}.txt') # 정상 라벨 0 을 정상 데이터 개수 만큼 생성
    y += [1] * (len(x) - len(y)) # 비정상 라벨 1을 비정상 데이터 개수 만큼 생성
    
    return x, y


def vectorize(train_x,test_x): # 문장을 벡터로 만듭니다 해당 코드에서는 기본적인 tf idf를 사용하고 있습니다.
    tf = TfidfVectorizer()
    tf = tf.fit(train_x)

    train_vec = tf.transform(train_x)
    test_vec = tf.transform(test_x)

    return train_vec,test_vec


def train(train_vec, train_y, select): # 랜덤 포레스트로 훈련 시킵니다. 모델을 바꾸고 싶다면 이 함수를 변경해야 합니다.
    if select == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=350)
        model.fit(train_vec, train_y)

    elif select == "SVM":
        model = LinearSVC(C=1)
        model.fit(train_vec, train_y)

    elif select == "Decision Tree":
        model = DecisionTreeClassifier(random_state=156)
        model.fit(train_vec, train_y)

    elif select == "LigthGBM":
        model = LGBMClassifier(learning_rate=0.1, 
                                n_estimators=1000, 
                                max_bin=300,
                                num_leaves=35)
        model.fit(train_vec, train_y)
        # boosting_type은 goss, dart보다 default(gbdt)가 좋았음
        # learning_rate와 n_estimators는 각각 0.1, 1000이 가장 높은 Acc
        # max_bin과 num_leaves는 미세하게 조정했을 때, 살짝의 Acc 향상이 있었음..
        # max_depth, min_data_in_leaf는 조정 안하는 것이 가장 나았다.

    elif select == "Logistic":
        model = LogisticRegression(max_iter=200, C=5)
        model.fit(train_vec, train_y)
        # max_iter 값은 200 이상부터는 Acc가 거의 동일함 200, 1000, 2000 등
        # cost function C 1) C=1 Acc: 0.9760091705559649
        #                 2) C=5 Acc: 0.9914844837468272 가 가장 적당할 듯
        #                 3) C=10 Acc: 0.9930402030623107

    elif select == "SGD":
        model = SGDClassifier(loss='perceptron', alpha=0.00001)
        model.fit(train_vec, train_y)
        # loss function 1) 확률적 경사하강법 적용 퍼셉트론 perceoptron Acc: 0.9883730451158601
        #               2) 확률적 경사하강법 적용 SVM hinge Acc: 0.9759272905919921
        #               3) 확률적 경사하강법 적용 로지스틱 회귀 log Acc: 0.9647916154916892

        # alpha 1) 0.00001 Acc: 0.9912388438549087
        #       2) 0.0005 Acc: 0.9838696470973552
        #       3) 0.0001 Acc: 0.9873086055842135
        #       4) 0.001 Acc: 0.9810857283222796

    return model


def test(test_y, test_vec, output, model): # 입렵 받은 테스트와 모델로 테스트를 실시합니다
    pred = output.predict(test_vec)

    print("")
    print(model, "Acc:", accuracy_score(test_y, pred))
    print(model, "F1 score :", f1_score(test_y, pred))

    # lightgbm plot_importance 코드
    # # 모델에서 feature가 사용된 횟수
    # ax = lgb.plot_importance(output, max_num_features=15, importance_type='split')
    # ax.set(title=f'Feature Importance (split)',
	#        xlabel='Feature Importance',
	#        ylabel='Features')

    # # total gains of splits which use the feature, 해당 feature의 상대적 기여도를 의미
    # ax = lgb.plot_importance(output, max_num_features=15, importance_type='gain')
    # ax.set(title=f'Feature Importance (gain)',
	#        xlabel='Feature Importance',
	#        ylabel='Features')
    # plt.show()

    return pred


def run():
    ############### 실행 코드 #######################
    train_x, train_y = dataset('./data/', 'train') # 경로 자기껄로 맞추기
    # print(len(train_x)) # 48852
    # print(len(train_y)) # 48852
    print("Success train dataset loading")
    test_x, test_y = dataset('./data/', 'test')
    print("Success test dataset loading")

    train_vec, test_vec = vectorize(train_x, test_x)

    # model selection
    model = ["Random Forest", "SVM", "Decision Tree", "LigthGBM", "Logistic", "SGD"]

    # Random Forest
    output0 = train(train_vec, train_y, model[0])
    pred0 = test(test_y, test_vec, output0, model[0])

    # SVM
    output1 = train(train_vec, train_y, model[1])
    pred1 = test(test_y, test_vec, output1, model[1])

    # Decision Tree
    output2 = train(train_vec, train_y, model[2])
    pred2 = test(test_y, test_vec, output2, model[2])

    # LigthGBM
    output3 = train(train_vec, train_y, model[3])
    pred3 = test(test_y, test_vec, output3, model[3])

    # Logistic
    output4 = train(train_vec, train_y, model[4])
    pred4 = test(test_y, test_vec, output4, model[4])

    # SGD
    output5 = train(train_vec, train_y, model[5])
    pred5 = test(test_y, test_vec, output5, model[5])

    ################ 실행 결과 #######################

    '''
    Random Forest Acc: 0.9647916154916892
    Random Forest F1 score : 0.9581874756903929

    SVM Acc: 0.9945959223777942
    SVM F1 score : 0.993382795267696

    Decision Tree Acc: 0.9671661344469008
    Decision Tree F1 score : 0.9608053953670218

    LigthGBM Acc: 0.9694587734381397
    LigthGBM F1 score : 0.9634922188509347

    Logistic Acc: 0.9914844837468272
    Logistic F1 score : 0.9895351177299255

    SGD Acc: 0.990665684107099
    SGD F1 score : 0.9885265700483091
    '''

if __name__=="__main__":
    run()