import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

def get_iris(mode=None):
    # 데이터 읽기
    iris = pd.read_csv('iris.csv')

    # 필요없는 컬럼 제거
    df = iris.drop(['Id'], axis=1).copy()

    # 컬럼명 변경
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    # 이진 데이터
    if (mode == 'bin'):
        df = df.loc[df['species'] != 'Iris-virginica']
  
    # 인코딩
    df['species'] = df['species'].map({'Iris-setosa': 0, 
                                       'Iris-versicolor': 1, 
                                       'Iris-virginica': 2})

    # X, y 분리
    X = df.drop(['species'], axis=1)
    y = df['species']

    # train, test 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    return X_train, X_test, y_train, y_test

def print_score(y_true, y_pred, average='binary'):
    # 정확도
    acc = accuracy_score(y_true, y_pred)
    # 정밀도
    pre = precision_score(y_true, y_pred, average=average)
    # 재현율
    rec = recall_score(y_true, y_pred, average=average)

    print('accuracy:', acc)
    print('precision:', pre)
    print('recall:', rec)

def plot_confusion_matrix(y_true, y_pred):
    # 혼동 행렬
    cfm = confusion_matrix(y_true, y_pred)
    # 시각화
    plt.figure(figsize=(5, 5))
    sns.heatmap(cfm, annot=True, cbar=False, fmt='d', cmap='coolwarm')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()