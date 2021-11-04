import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  #랜덤포레스트 모델만 알고 있어 랜덤 포레스트 모드로 진행하였습니다
from sklearn.metrics import accuracy_score,f1_score

def parsing(path):#파싱을 진행하는 함수
    with open(path,'r',encoding='utf-8') as f:#파일을 읽어드리고 ['로그','로그',...] 이런식으로 로그를 구조화
        train=[]
        para=""
        while True:
            l = f.readline() #한줄씩 읽어 옵니다

            if not l:
                break #파일을 전부 읽으면 읽기를 중단합니다.

            if l != "\n":
                if l[:3] == 'GET' or l[:4] == 'POST':
                    para += l
            else:
                if para!='':
                    if para[:4]=='POST': #Method가 POST인 경우 예외적으로 바디까지 가져옵니다.
                        para+=f.readline()
                    train.append(para)
                    para=""
    return train

def dataset(path,mod='train'): #데이터셋을 생성합니다. 파싱한 데이터와 라벨을 생성합니다 
    x = parsing(f'{path}norm_{mod}.txt') # mod에 따라 train을 가져올지 test 데이터를 가져올지 결정됩니다.
    y = [0]*len(x) # 정상 라벨 0 을 정상 데이터 개수 만큼 생성
    x += parsing(f'{path}anomal_{mod}.txt')
    y += [1]*(len(x)-len(y)) # 비정상 라벨 1을 비정상 데이터 개수 만큼 생성
    return x, y

def vectorize(train_x,test_x): #문장을 벡터로 만듭니다 해당 코드에서는 기본적인 tf idf를 사용하고 있습니다.
    tf = TfidfVectorizer()
    tf = tf.fit(train_x)
    train_vec = tf.transform(train_x)
    test_vec = tf.transform(test_x)
    return train_vec,test_vec

def train(train_vec,train_y): #랜덤 포레스트로 훈련 시킵니다. 모델을 바꾸고 싶다면 이 함수를 변경해야 합니다.
    rf = RandomForestClassifier()
    rf.fit(train_vec,train_y)
    return rf

def test(test_y,test_vec,rf): #입렵 받은 테스트와 모델로 테스트를 실시합니다
    pred = rf.predict(test_vec)
    print("Accuracy" + accuracy_score(test_y,pred))
    print("f1_score" + f1_score(test_y,pred))
    return pred


def run():
############### 실행 코드 #######################
################################################

    train_x, train_y = dataset('./','train')
    test_x, test_y =  dataset('./','test')

    train_vec, test_vec = vectorize(train_x, test_x)
    rf = train(train_vec, train_y)
    pred = test(test_y,test_vec, rf)
    
########################################################
    tf = TfidfVectorizer()
    tf = tf.fit(train_x)

    # print(len(tf.vocabulary_))

    # print(tf.transform(train_x)[0])


if __name__=="__main__":
    run()   
