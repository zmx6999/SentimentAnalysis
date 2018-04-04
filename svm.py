import pandas as pd
import jieba
import numpy as np
from sklearn import model_selection,svm
from gensim.models import word2vec
import sklearn.externals.joblib as joblib
def loadfile():
    pos = pd.read_excel('pos.xls',header=None)
    neg = pd.read_excel('neg.xls',header=None)
    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)
    x = np.concatenate((pos['words'],neg['words']))
    y = np.concatenate((np.ones(len(pos['words'])),np.zeros(len(neg['words']))))
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=.2)
    np.save('svm_y_train.npy',y_train)
    np.save('svm_y_test.npy',y_test)
    return x_train,x_test


def build_word_vector(x,size,w2v):
    v = np.zeros((1,size))
    n = 0
    for w in x:
        try:
            v += w2v[w].reshape((1,size))
            n += 1
        except KeyError:
            continue
    if n > 0: v /= n
    return v


def get_train_vector(x_train,x_test):
    n_dim = 300
    w2v = word2vec.Word2Vec(size=n_dim,min_count=10)
    w2v.build_vocab(x_train)
    w2v.train(x_train,total_examples=len(x_train),epochs=w2v.iter)
    train_v = np.concatenate([build_word_vector(x,n_dim,w2v) for x in x_train])
    np.save('svm_train_v.npy',train_v)

    w2v.train(x_test,total_examples=len(x_train),epochs=w2v.iter)
    test_v = np.concatenate([build_word_vector(x,n_dim,w2v) for x in x_test])
    np.save('svm_test_v.npy',test_v)
    w2v.save('svm_w2v_model.pkl')


def svm_train(train_v,y_train,test_v,y_test):
    svc = svm.SVC(kernel='rbf')
    svc.fit(train_v,y_train)
    joblib.dump(svc,'svm_model.pkl')
    print(svc.score(test_v,y_test))


def get_predict_vector(words):
    n_dim = 300
    w2v = word2vec.Word2Vec.load('svm_w2v_model.pkl')
    return build_word_vector(words,n_dim,w2v)


def svm_predict(str):
    words = jieba.lcut(str)
    v = get_predict_vector(words)
    svm = joblib.load('svm_model.pkl')
    print(svm.predict(v))


if __name__ == '__main__':

    x_train,x_test = loadfile()
    get_train_vector(x_train,x_test)

    train_v = np.load('svm_train_v.npy')
    y_train = np.load('svm_y_train.npy')
    test_v = np.load('svm_test_v.npy')
    y_test = np.load('svm_y_test.npy')

    svm_train(train_v,y_train,test_v,y_test)

    str = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    svm_predict(str)
    str = '牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    svm_predict(str)