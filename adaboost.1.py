import numpy as np
import matplotlib.pyplot as plt
from sklearn import __version__ as sklearn_version
if sklearn_version < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

from sklearn import tree

def main():

    #データのロード
    iris_data = np.loadtxt("data/iris.data.txt",delimiter=",",usecols=(0,1,2,3))
    iris_target = np.loadtxt("data/iris.data.txt",delimiter=",",usecols=(4), dtype=np.dtype("U16"))
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    target_names = np.array(['setosa', 'versicolor', 'virginica'])
    X = np.array(iris_data[:,[0,2]])  # sepal length and petal length
    y = []
    for i in range(len(iris_target)):
        if(iris_target[i] == 'Iris-versicolor'):
            y.append(1)
        elif(iris_target[i] == 'Iris-virginica'):
            y.append(-1)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    boost = Adaboost(1000)
    boost.fit(X_train,y_train)
    score = boost.score(X_test, y_test)
    print('最終テスト結果：正解率={0}'.format(score))
    
#弱学習器クラス
class WeakClassifier(object):

    def __init__(self):
        #最大深さ
        self.max_depth = 5
        #乱数シード
        self.random_state = 3
        #決定木
        self.weak_cls = tree.DecisionTreeClassifier(criterion="gini", max_depth=self.max_depth, random_state=self.random_state)
        
        #重み付けトレーニングセット
        self.X_train_weight = None
        #重み付けトレーニングセット
        self.y_train_weight = None

        #誤差率
        self.error_rate = None
        #信頼度
        self.reliability = None

    def learn(self,X_train, y_train, weight):
        #データ重みに従い、トレーニングセットをサンプリング
        (X_train_sample , y_train_sample) = self.randomsampling(X_train, y_train, weight)

        #学習
        self.weak_cls.fit(X_train_sample, y_train_sample)
        
        #誤差率、信頼度を計算
        y_predicted = self.weak_cls.predict(X_train)
        result = np.array([])
        for i in range(y_train.shape[0]):
            if(y_predicted[i] != y_train[i]):
                #error
                result = np.append(result,1)
            else:
                result = np.append(result,0)

        self.error_rate = np.dot(result,weight) / np.sum(weight)
        self.reliability = np.log((1 - self.error_rate)/self.error_rate)

        #重みを更新
        weight_new = np.array([])
        for i in range(weight.shape[0]):
            #error
            if result[i] == 1:
                weight_new = np.append(weight_new, weight[i] * np.exp(self.reliability))
            else:
                weight_new = np.append(weight_new, weight[i])
        #新しい重みを応答
        return weight_new / np.sum(weight_new)

    #重み付けランダムサンプリング
    #重みに応じて、インプットデータと同じ数だけサンプリングし出力
    def randomsampling(self,X_train, y_train, weight):
        y_num = y_train.shape[0]
        w_num = weight.shape[0]
        if y_num != w_num:
            return
        
        weight_normalized = weight / np.sum(weight)
        X_result = []
        y_result = []
        for i in range(y_num):
            b = np.random.rand(1)[0]
            a = 0
            for j in range(w_num):
                a += weight_normalized[j]
                if(a >= b):
                    X_result.append(X_train[j,:].tolist())
                    y_result.append(y_train[j].tolist())
                    break
        return (np.array(X_result) , np.array(y_result))
        
class Adaboost(object):
    #コンストラクター
    def __init__(self,num_wcls=100):
        #作成する弱学習器の数
        self.num_wcls = num_wcls

        #弱学習器たち
        self.wclss = []
        #信頼度
        self.reliabilities = []

        print('Adaboost  弱学習器の数 : {0}'.format(self.num_wcls))

    #boosting学習
    def fit(self,X_train, y_train):
        #初期重み付け
        weight_0 = np.ones(y_train.shape[0])  / y_train.shape[0]
        weight_next = None
        #弱学習器の作成⇒学習
        for i in range(self.num_wcls):
            wcls = WeakClassifier()
            if i == 0:
                weight_next = wcls.learn(X_train,y_train,weight_0)
            else:
                weight_next = wcls.learn(X_train,y_train,weight_next)
            self.wclss.append(wcls)
            self.reliabilities.append(wcls.reliability)

            print('弱学習器: {0}  エラー率: {1}'.format(i, wcls.error_rate))
            print('弱学習器: {0}  信頼度: {1}'.format(i, wcls.reliability))
    
    #予測
    def predict(self,X_predict):
        #強学習器による予測
        y_predict = np.zeros(X_predict.shape[0])
        for i in range(len(self.wclss)):
            y_predict += self.reliabilities[i] * self.wclss[i].weak_cls.predict(X_predict)

        for j in range(y_predict.shape[0]):
            if y_predict[j] > 0:
                y_predict[j] = 1
            else:
                y_predict[j] = -1

        return y_predict

    #予測し、正解率を応答
    def score(self, X_predict , y_result):
        #予測
        y_predict = self.predict(X_predict)
        num_correct = 0
        for j in range(y_predict.shape[0]):
            if y_predict[j] * y_result[j] > 0:
                num_correct += 1
            else:
                pass
        
        return num_correct / y_predict.shape[0]

if __name__ == "__main__":
    main()