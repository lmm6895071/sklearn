# -*- coding:utf-8 -*-

import sys
import os
reload(sys)
sys.setdefaultencoding('utf8')

import re
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from numpy import *
from sklearn import metrics
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
import time
import datetime

from sklearn.neural_network import MLPClassifier

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

f=open(dirname+"/stopword.txt","r")
ls=f.read()
f.close()
lss=ls.split("\n")
STOPWORD=[]
for item in lss:
    item=item.strip("\n")
    item=item.strip(' ')
    STOPWORD.append(item)
GLOBALVECTOR=[]
GLOBALWORDS=[]
class VModel:
    posCount=0
    negCount=0
    def __init__(self):
        pass
        self.model = gensim.models.Word2Vec.load(dirname+"/outmodle/wiki.ch.text.model")

    def cleanText(self, documents):
        result = []
        for document in documents:
            document = document.replace('\n', '')
            words = []
            for word in document.split():
                word = unicode(word, "utf-8")
                words.append(word)
            result.append(words)
        return result

    def getPLData(self,posCount,negCount):
        path = "f_corpus/"
        #fname = "ch_waimai_corpus.txt"
        #fname = "ch_waimai2_corpus.txt"
        fname = "ch_hotel_corpus.txt"
        infile = open(path + fname)
        posD=[]
        negD=[]
        rdata1=[]
        rdata2=[]
        for line in infile.readlines():
            if line[0:3] == "neg":
                negD.append(line[4:].strip().strip("\n"))
            else:
                posD.append(line[4:].strip().strip("\n"))

        print "pos counts:",len(posD)
        print "neg counts:",len(negD)

        if posCount > 0 :
            shuffleArray = range(len(posD))
            np.random.shuffle(shuffleArray)
            for ii in xrange(posCount):
                rdata1.append(posD[shuffleArray[ii]])
        else:
            rdata1 = posD

        if negCount > 0:
            shuffleArray = range(len(negD))
            np.random.shuffle(shuffleArray)
            for ii in xrange(negCount):
                rdata2.append(negD[shuffleArray[ii]])
        else:
            rdata2 = negD
        return  (rdata1,rdata2)
    def getRawDocument(self):
        rdata2 = []
        rdata1 = []
        print 'start------------------'
        print dirname
        '''
        with open(dirname+"/data/trainData/trans_neg.txt") as infile:
            data2 = infile.readlines()

        with open(dirname+"/data/trainData/trans_pos.txt") as infile:
            data1 = infile.readlines()
        '''
        with open("fupeng/cut_neg.txt") as infile:
            data2 = infile.readlines()

        with open("fupeng/cut_pos.txt") as infile:
            data1 = infile.readlines()

        VModel.posCount = len(data2)
        VModel.negCount = len(data2)

        shuffleArray = range(len(data1))
        np.random.shuffle(shuffleArray)
        for ii in xrange(VModel.posCount):
            rdata1.append(data1[shuffleArray[ii]])

        shuffleArray = range(len(data2))
        np.random.shuffle(shuffleArray)
        for ii in xrange(VModel.negCount):
            rdata2.append(data2[shuffleArray[ii]])
        return (rdata1,rdata2)
    def getFeature(self,X):
        '''vectorrize=CountVectorizer()
        texts =[]
        for text in X:
            str=''
            for word in text:
                str=str+ word+" "
                texts.append(str)
        print len(texts)
        XX =vectorrize.fit_transform(texts)
        word=vectorrize.get_feature_names()'''
        '''transformor = TfidfTransformer()
        print transformor
        tfidf = transformor.fit_transform(XX)
        weight = tfidf.toarray()
        print np.shape(weight)
        weight = weight.tolist()
        weight = scale(weight)'''
        word=[]
        for item in X:
            for w in item:
                if w !='' or w !=" " or w !="\n":
                    word.append(w)
        fw=open("features1.txt","w")
        print len(word)
        cc = 0
        GLOBALWORDS=[]
        STOPWORDSSET=set(STOPWORD)
        for w in word:
            w=w.strip(' ')
            w=w.strip("\n")
            w=re.sub("\d","",w)
            if w not in STOPWORDSSET and w !='' and w!=' ' and len(w)>1:
                try:
                    cc=cc+1
                    print cc ,'\t',w
                    GLOBALWORDS.append(w)
                except:
                    continue
        print "init count ",
        print len(GLOBALWORDS)
        GLOBALWORDS = list(set(GLOBALWORDS))
        print len(GLOBALWORDS)
        for w in GLOBALWORDS:
            fw.write(w+"\n")
        fw.close()
        print cc
    def getKMSModel(self,sz=1500):
        fw=open("features.txt","r")
        words=fw.readlines()
        fw.close()
        CC =0
        GLOBALVECTOR=[]
        GLOBALWORDS=[]
        print len(words)
        for item in words:
            if item =="":
                continue
            item=item.strip(" ")
            item= item.strip("\n")
            print "item:",item
            try:

                item=unicode(item,"utf-8")
                vec=self.model[item]
                CC=CC+1
                GLOBALVECTOR.append(vec)
                GLOBALWORDS.append(item)
                print CC
            except Exception as err:
                print item
                continue
        print "end"
        print len(GLOBALWORDS)
        print len(GLOBALVECTOR)
        clf = KMeans(n_clusters=sz)
        s = clf.fit(GLOBALVECTOR)
        print s
        lables= clf.labels_
        joblib.dump(clf , 'kms_1500.pkl')
        print "lables length:",len(lables)
        print "word count:",len(GLOBALWORDS)
        print x,'----------------------',clf.inertia_
        fl=open("lables_1500.txt","w")
        lables=lables.tolist()
        for itm in range(len(GLOBALWORDS)):
            fl.write(GLOBALWORDS[itm]+"\t")
            s=unicode(lables[itm])
            s=s+"\n"
            fl.write(s)
        fl.close()

    def getInputData(self,size=400):
        (pos_data,neg_data)=self.getRawDocument()
        fs = open("lables_2000.txt","r")
        features = fs.readlines()
        fs.close()
        labs = {}
        for item in features:
            item = item.strip("\t")
            item = item.split("\t")
            if len(item)>1:
                k= unicode(item[0],'utf-8')
                v=unicode(item[1],'utf-8')
                labs[k] = v
        Fword = labs.keys()
        print "labs word count:", len(Fword)
        result = list([])
        datas =pos_data + neg_data
        print len(pos_data), len(neg_data)
        print "datas length",len(datas)
        for item in datas:
            doc=np.zeros(size)
            doc=doc.tolist()
            words = item.split(' ')
            for w in words:
                w=w.strip("\n")
                if w == '':
                    continue
                try:
                    w = unicode(w,'utf-8')
                    value = int(labs[w])
                    doc[value-1] = doc[value-1]+1
                except:
                    #print "skip ",w
                    continue
            result.append(doc)
        print len(result)
        print "total count ",len(datas)
        y = np.concatenate((np.ones(len(pos_data)), np.zeros(len(neg_data))))
        print "end ------------",len(result), len(y)
        return (result,y)
    def getWordVector(self,word):
        vec = np.zeros(size).reshape(1,size)
        print word
        vec = self.model[word]
    def buildWordVector(self,words,size=400):
        vec = np.zeros(size).reshape((1,size))
        count = 0
        for word in words:
            try:
                vec += self.model[word].reshape((1,size))
                count += 1
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec
    def testModel(self,X,y,size=0.2):
        classifiers = [
            KNeighborsClassifier(),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            LogisticRegression(),
            GradientBoostingClassifier(),
            MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
        ]
        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
             "Decision Tree", "Random Forest", "AdaBoost",
             "LogisticRegression", "GradientBoostingClassifier","MLPClassifier"]

        #classifiers = [clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(5, 2), random_state=1)  ()]
        #names=["LogisticRegression"]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size)


        for name,clf in zip(names,classifiers):
            d1 = datetime.datetime.now()

            print name
            startT = time.time()
            clf.fit(X_train,y_train)
            endT = time.time()
            y_true,y_pred = y_test,clf.predict(X_test)
            startT = time.time()
            #print (clf.coef_)
            print (classification_report(y_true,y_pred))
            print (metrics.confusion_matrix(y_true,y_pred))

            d2 = datetime.datetime.now()
            interval=d2-d1

            print "===========time============",interval.days*24*3600 + interval.seconds+interval.microseconds/1000000.0

def testVector():
    d1 = datetime.datetime.now()
    model = VModel()
    #pos_data,neg_data=model.getRawDocument()
    pos_data,neg_data=model.getPLData(1000,1000)
    print VModel.posCount, VModel.negCount

    y = np.concatenate((np.ones(len(pos_data)), np.zeros(len(neg_data))))
    X = np.concatenate((pos_data, neg_data))
    X = model.cleanText(X)
    X = np.concatenate([model.buildWordVector(z) for z in X])
    X = scale(X)
    X_vec = []
    for item in X:
        X_vec.append(tuple(item.tolist()))

    d2 = datetime.datetime.now()
    interval=d2-d1

    print "yuchuli time ,",interval.days*24*3600 + interval.seconds+interval.microseconds/1000000.0

    model.testModel(X_vec,y)

def testVectorKMSLable():
    model = VModel()
    (x,y)=model.getInputData(2000)
    model.testModel(x,y)

def productKMSLables():

    model = VModel()
    '''
    pos_data,neg_data=model.getRawDocument()
    print VModel.posCount, VModel.negCount
    y = np.concatenate((np.ones(len(pos_data)), np.zeros(len(neg_data))))
    X = np.concatenate((pos_data, neg_data))
    X = model.cleanText(X)
    '''
    #model.getFeature(X)
    model.getKMSModel()

if __name__ == "__main__":
    testVector()
    #productKMSLables()
    #testVectorKMSLable()



