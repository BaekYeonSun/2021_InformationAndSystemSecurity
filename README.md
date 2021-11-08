#  정보보호와시스템보안
## 1. CSIC 2010 Dataset
-----

### 팀원 소개
```
20191604 백연선
GitHub : https://github.com/BaekYeonSun
```
```
20191670 조나영
GitHub : https://github.com/NaYoung2253
```
```
20202096 원채은
GitHub : https://github.com/chaeeun3
```
```
20191313 채지윤
GitHub : https://github.com/jiyoony
```

-----
### HTTP DATASET CSIC 2010

    HTTP DATASET CSIC 2010에는 자동으로 생성된 수천 개의 웹 요청이 포함되어 있습니다. 
    웹 공격 방지 시스템 테스트에 사용할 수 있습니다. CSIC의 정보 보안 연구소에서 개발되었습니다.
    
출처 - https://www.tic.itefi.csic.es/dataset/

-----
### 머신러닝 알고리즘

    머신러닝은 지도 학습(Supervised Learning), 비지도 학습(Unsupervised Learning), 
    강화 학습(Reinforcement Learning)으로 구분됩니다.

    그 중 지도 학습과 비지도 학습의 차이점은 학습 결과에 대한 사전 지식의 존재 유무입니다. 
    지도 학습은 학습 결과에 대한 사전 지식이 존재하므로 분류(classification)와 회귀(regression)가 가능하고, 
    비지도 학습은 학습 결과에 대한 사전 지식이 존재하지 않으므로 비슷한 데이터들을 군집화 하여 결과를 예측합니다.  


이번 프로젝트에서 활용한 지도 학습의 알고리즘은 다음과 같습니다.

1. **Logistic Regression(로지스틱 회귀)**
    
    로지스틱 회귀의 목적은 일반적인 회귀 분석의 목표와 동일하게 종속 변수와 독립 변수간의 관계를 구체적인 함수로 나타내어 향후 예측 모델에 사용하는 것입니다.
    
    하지만 로지스틱 회귀는 선형 회귀 분석과는 다르게 종속 변수가 범주형 데이터를 대상으로 하며 입력 데이터가 주어졌을 때 해당 데이터의 결과가 특정 분류로 나뉘기 때문에 일종의 분류 (classification) 기법으로도 볼 수 있습니다.

2. **Decision Tree(결정 트리)**

    결정 트리는 데이터를 분석해서 데이터 사이에 존재하는 패턴을 예측 가능한 규칙의 조합으로 나타내는 것입니다. 결정 트리는 분류와 회귀 모두 가능합니다. 즉, 범주나 연속형 수치 모두 예측할 수 있습니다.

3. **Random forests(랜덤 포레스트)**

    랜덤 포레스트는 훈련 과정에서 구성한 다수의 결정 트리로부터 부류(분류) 또는 평균 예측치(회귀 분석)를 출력합니다.

4. **LightGBM(Gradient Boosting Machine, 점진적 부스팅 머신)**

    LightGBM은 일반적인 균형 트리 분할 방식(Level Wise)과 다르게 리프 중심 트리 분할 방식(Leaf Wise)을 이용합니다. 이 방법은 트리의 균형을 맞추지 않고 최대 손실 값(max delta loss)을 가지는 리프 노드를 지속적으로 분할하면서 트리가 깊어지고 비대칭 적인 트리를 만듭니다. 
    
    하지만 이 방법으로 트리를 계속해서 분할하게 되면 균형 트리 분할 방식보다 예측 오류 손실을 최소화할 수 있습니다.

5. **SVM(Support Vector Machine, 서포트 벡터 머신)**

    SVM은 주어진 데이터 집합을 바탕으로 하여 새로운 데이터가 어느 카테고리에 속할지 판단하는 비확률적 이진 선형 분류 모델을 만듭니다. 만들어진 분류 모델은 데이터가 사상된 공간에서 경계로 표현되는데 그 중 가장 큰 폭을 가진 경계를 찾는 알고리즘입니다.

6. **SGD(Stochastic Gradient Descent, 확률적 경사 하강법)**

    확률적 경사 하강법은 추출된 데이터 한개에 대해서 gradient를 계산하고 경사 하강 알고리즘을 적용하는 방법입니다. 전체 데이터를 사용하는 것이 아닌 랜덤하게 추출한 일부 데이터만을 사용하므로 학습 중간 과정에서 결과의 진폭이 크고 불안정하고 속도가 매우 빠르다. 또한, 데이터를 하나씩 처리하기 때문에 오차율이 크고 GPU의 성능을 모두 활용하지 못합니다. 

-----
### 용어 정리
- Accuracy(정확도) - 올바르게 예측된 데이터의 수를 전체 데이터의 수로 나눈 값
- F1 Score - precision과 recall의 조화평균
    - Precision(정밀도) - 모델이 True로 예측한 데이터 중 실제로 True인 데이터의 수
    - Recall(재현율) - 실제로 True인 데이터를 모델이 True라고 인식한 데이터의 수

-----
### 코드 설명

```python
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
```

```python
def dataset(path, mod='train'):
    x = parsing(f'{path}norm_{mod}.txt') # 데이터셋을 생성합니다. 파싱한 데이터와 라벨을 생성합니다 
    y = [0] * len(x) # 데이터셋을 생성합니다. 파싱한 데이터와 라벨을 생성합니다 
    
    x += parsing(f'{path}anomal_{mod}.txt') # 정상 라벨 0 을 정상 데이터 개수 만큼 생성
    y += [1] * (len(x) - len(y)) # 비정상 라벨 1을 비정상 데이터 개수 만큼 생성
    
    return x, y
```

```python
def vectorize(train_x,test_x): # 문장을 벡터로 만듭니다 해당 코드에서는 기본적인 tf idf를 사용하고 있습니다.
    tf = TfidfVectorizer()
    tf = tf.fit(train_x)

    train_vec = tf.transform(train_x)
    test_vec = tf.transform(test_x)

    return train_vec,test_vec
```

```python
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
```

```python
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
```

+) lightgbm의 plot_importance 그래프
<div>
    <img width="300px" src="https://user-images.githubusercontent.com/55417591/140703432-22536e16-8330-4ad7-87b8-3615effc990e.png"/>
    <img width="300px" src="https://user-images.githubusercontent.com/55417591/140703477-350f3973-a453-41d9-bc6d-bf82ab2b5593.png"/>
</div>

```python
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
```

-----
### 실행 결과
``` 
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
```
