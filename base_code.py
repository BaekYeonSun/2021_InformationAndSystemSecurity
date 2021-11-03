# import pandas as pd
# import re
# import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score


def parsing(path):
    with open(path, 'r', encoding='utf-8') as f: # 파일 열기
        data = []
        para = ""
        
        line = "parsing start"
        while line:
            line = f.readline() # 한 줄씩 read
                
            if line.startswith("GET"): # GET 부분
                para += line
#                 print("Get/", para)
                data.append(para)
                para = ""
            
            # POST and PUT 부분
            elif line.startswith("POST") or line.startswith("PUT"):
                para += (line + " ")
                l = ""
                while not l.startswith("Content-Length"):
                    l = f.readline()
#                     print("Post/", l)
                l = f.readline()
                l = f.readline()
                para += l
                data.append(para)
                para = ""
        
#     print(data)
                    
    return data


def dataset(path, mod='train'):
    x = parsing(f'{path}norm_{mod}.txt')
    y = [0] * len(x)
    
    x += parsing(f'{path}anomal_{mod}.txt')
    y += [1] * (len(x) - len(y))
    
    return x, y


def vectorize(train_x,test_x):
    tf = TfidfVectorizer()
    tf = tf.fit(train_x)

    train_vec = tf.transform(train_x)
    test_vec = tf.transform(test_x)

    return train_vec,test_vec


def train(train_vec, train_y, select):
    if select == "Random Forest":
        model = RandomForestClassifier()
        model.fit(train_vec, train_y)
    elif select == "SVM":
        model = LinearSVC(C=1)
        model.fit(train_vec, train_y)
    elif select == "Decision Tree":
        model = DecisionTreeClassifier()
        model.fit(train_vec, train_y)
    elif select == "LigthGBM":
        model = LGBMClassifier()
        model.fit(train_vec, train_y)
    elif select == "Logistic":
        model = LogisticRegression(max_iter=1000)
        model.fit(train_vec, train_y)
    elif select == "SGD":
        model = SGDClassifier()
        model.fit(train_vec, train_y)

    return model


def test(test_y, test_vec, output, model):
    pred = output.predict(test_vec)

    print("")
    print(model, "Acc:", accuracy_score(test_y, pred))
    print(model, "F1 socre :", f1_score(test_y, pred))

    return pred


def run():
    ############### 실행 코드 #######################
    train_x, train_y = dataset('', 'train') # 경로 자기껄로 맞추기
    print("Success train dataset loading")
    test_x, test_y = dataset('', 'test')
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
    Random Forest Acc: 0.9640546958159338
    Random Forest F1 socre : 0.9572915653273664

    SVM Acc: 0.9945959223777942
    SVM F1 socre : 0.993382795267696

    Decision Tree Acc: 0.9656104151314173
    Decision Tree F1 socre : 0.95892822217876

    LigthGBM Acc: 0.9558666994186522
    LigthGBM F1 socre : 0.9460298387904276

    Logistic Acc: 0.9760091705559649
    Logistic F1 socre : 0.9705557230429103

    SGD Acc: 0.9760091705559649
    SGD F1 socre : 0.9702930142958531
    '''

    ################ (+++) #######################
    # 모델만 바꿔도 Acc가 99가 나와서.. 앞으로 뭘 해야할지 생각해봤는데
    # 1) 99가 안나온 모델의 파라미터를 바꾸거나 수정하면서 Acc를 높여도 될 것 같고,
    # 2) 혹은 내가 더 좋은 모델이 있어서 추가하고 싶다면 추가해서 결과를 봐도 좋을 것 같다.
    # 3) README.md 파일을 통해서 사용한 모델은 어떤 모델인지 간략하게 설명하고, 왜 사용하게 됐는지..?를 얘기해도 좋을 듯..
    # 4) Acc 99 나왔으니 그만하자..
    # 5) 딥러닝 도전...???

if __name__=="__main__":
    run()