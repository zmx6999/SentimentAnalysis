import pandas
import numpy
import jieba
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import multiprocessing
from keras import models,layers,optimizers
from keras.preprocessing import sequence
from sklearn import model_selection
import yaml
vocab_dim = 100


def loadfile():
    pos = pandas.read_excel('pos.xls',header=None)
    neg = pandas.read_excel('neg.xls',header=None)
    x = numpy.concatenate((pos[0],neg[0]))
    y = numpy.concatenate((numpy.ones(len(pos)),numpy.zeros(len(neg))))
    return x,y


def tokenizer(x):
    return [jieba.lcut(v.replace('\n','')) for v in x]


def get_word_vector(x):
    w2v = Word2Vec(size=vocab_dim,min_count=10,window=7,workers=multiprocessing.cpu_count(),iter=1)
    w2v.build_vocab(x)
    w2v.train(x,total_examples=len(x),epochs=w2v.iter)
    w2v.save('w2v.pkl')
    return get_word_dictionary(x,w2v)


def get_word_dictionary(x,w2v):
    d = Dictionary()
    d.doc2bow(w2v.wv.vocab.keys(),allow_update=True)
    w2v_index = {v:k+1 for k,v in d.items()}
    w2vec = {w:w2v[w] for w in w2v_index}

    def get_data_vector(x):
        data = []
        for v in x:
            vv = []
            for w in v:
                try:
                    vv.append(w2v_index[w])
                except:
                    vv.append(0)
            data.append(vv)
        return data

    x = get_data_vector(x)
    x = sequence.pad_sequences(x,maxlen=100)
    return w2v_index,w2vec,x


def get_data(w2v_index,w2vec,x,y):
    n = len(w2v_index)+1
    weights = numpy.zeros((n,vocab_dim))
    for w in w2v_index:
        weights[w2v_index[w]] = w2vec[w]
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=.2)
    return n,weights,x_train,x_test,y_train,y_test


def train_lstm(n,weights,x_train,x_test,y_train,y_test):
    sequential = models.Sequential()
    sequential.add(layers.Embedding(input_dim=n,output_dim=vocab_dim,input_length=100,mask_zero=True,weights=[weights]))
    sequential.add(layers.LSTM(50,activation='sigmoid',inner_activation='hard_sigmoid'))
    sequential.add(layers.Dropout(.5))
    sequential.add(layers.Dense(1,activation='sigmoid'))

    sequential.compile(optimizer=optimizers.adam(),loss='binary_crossentropy',metrics=['accuracy'])
    sequential.fit(x_train,y_train,epochs=4,validation_data=(x_test,y_test))

    yml_str = sequential.to_yaml()
    with open('lstm.yml','w') as f:
        f.write(yaml.dump(yml_str))
    sequential.save_weights('weights.h5')


def predict_lstm(x):
    x = tokenizer([x])
    w2v = Word2Vec.load('w2v.pkl')
    _,_,x = get_word_dictionary(x,w2v)
    with open('lstm.yml') as f:
        yml_str = yaml.load(f)
    sequential = models.model_from_yaml(yml_str)
    sequential.load_weights('weights.h5')
    sequential.compile(optimizer=optimizers.adam(), loss='binary_crossentropy')
    return sequential.predict_classes(x)


if __name__ == '__main__':
    '''
    x,y = loadfile()
    x = tokenizer(x)
    w2v_index, w2vec, x = get_word_vector(x)
    n, weights, x_train, x_test, y_train, y_test = get_data(w2v_index,w2vec,x,y)
    train_lstm(n, weights, x_train, x_test, y_train, y_test)
    '''
    print(predict_lstm('电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'))
    print(predict_lstm('牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'))
    print(predict_lstm('酒店的环境非常好，价格也便宜，值得推荐'))
    print(predict_lstm('手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'))
    print(predict_lstm('屏幕较差，拍照也很粗糙。'))
    print(predict_lstm('质量不错，是正品 ，安装师傅也很好，才要了83元材料费'))
    print(predict_lstm('国企是傻逼'))
    print(predict_lstm('柳钢是傻逼'))
    print(predict_lstm('东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'))