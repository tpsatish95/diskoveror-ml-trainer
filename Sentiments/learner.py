'''
Copyright 2015 Serendio Inc.
Author - Satish Palaniappan

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
'''
__author__ = "Satish Palaniappan"


import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import pickle
# from processor.twokenize import *

import sys
sys.path.append("../")
from SocialFilter.TextFilter import Filter

class TextLearner(object):
    def __init__(self,data_path,model_path = "./",name = ""):
        self.name = name
        self.data_path = data_path
        self.model_path = model_path
        self.DesignMatrix = []
        self.TestMatrix = []
        self.X_train = []
        self.y_train = [] # not only train but general purpose too
        self.X_test = []
        self.y_test  = []
        self.y_pred = []
        self.vectorizer = None
        self.feature_names = None
        self.chi2 = None
        self.mlModel = None
        self.F = Filter()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.DesignMatrix = []
        self.TestMatrix = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test  = []
        self.y_pred = []
        self.vectorizer = None
        self.feature_names = None
        self.chi2 = None
        self.mlModel = None
        self.F = None

    def addModelDetails(self,model_p,name = ""):
        self.name = name
        self.model_path = model_p


    def load_data(self,TrTe = 0):               #TrTe => 0-Train  1-Test # returns the dimensions of vectors
        with open( self.data_path, 'rb') as f:
            if TrTe == 0:
                self.DesignMatrix = pickle.load(f)
                return len(self.DesignMatrix[1])
            if TrTe == 1:
                self.TestMatrix = pickle.load(f)
                return len(self.TestMatrix[1])

    def clearOld(self):
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test  = []
        self.y_pred = []
        self.vectorizer = None
        self.feature_names = None
        self.chi2 = None
        self.mlModel = None


    def process(self,text,default = 0):
        if default == 0:
            text = text.strip().lower().encode("utf-8")
        else:
            text = self.F.process(text)
        return text


    def loadXY(self,TrTe = 0,feature_index = 0,label_index = 1):     #TrTe => 0-Train  1-Test
        if TrTe == 0:
            for i in self.DesignMatrix:
                self.X_train.append(self.process(i[feature_index]))
                self.y_train.append(i[label_index])
            self.X_train = np.array(self.X_train)
            self.y_train = np.array(self.y_train)

        elif TrTe == 1:
            for i in self.TestMatrix:
                self.X_test.append(self.process(i[feature_index]))
                self.y_test.append(i[label_index])
            self.X_test = np.array(self.X_test)
            self.y_test = np.array(self.y_test)


    def featurizeXY(self,only_train = 1):      # Extracts Features
        sw = ['a', 'across', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'being', 'but', 'by', 'can', 'could', 'did', 'do', 'does', 'each', 'for', 'from', 'had', 'has', 'have', 'in', 'into', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'of', 'on', 'or', 'that', "that's", 'thats', 'the', 'there', "there's", 'theres', 'these', 'this', 'those', 'to', 'under', 'until', 'up', 'were', 'will', 'with', 'would']
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words=sw)
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.feature_names = self.vectorizer.get_feature_names()
        if only_train == 0:
            self.X_test = self.vectorizer.transform(self.X_test)


    def reduceDimension(self,only_train = 1, percent = 50):      # Reduce dimensions / self best of features
        n_samples, n_features = self.X_train.shape
        k = int(n_features*(percent/100))

        self.chi2 = SelectKBest(chi2, k=k)
        self.X_train = self.chi2.fit_transform(self.X_train, self.y_train)
        self.feature_names = [self.feature_names[i] for i in self.chi2.get_support(indices=True)]
        self.feature_names = np.asarray(self.feature_names)
        if only_train == 0:
            self.X_test = self.chi2.transform(self.X_test)


    def trainModel(self,Model = "default"):
        if Model == "default":
            self.mlModel = LinearSVR(loss='squared_epsilon_insensitive',dual=False, tol=1e-3)
        else:
            self.mlModel = Model
        self.mlModel.fit(self.X_train, self.y_train)


    def testModel(self,approx = 1):        # returns score ONLY
        self.y_pred = np.array(self.mlModel.predict(self.X_test))

        if approx == 1:
            ### To convert real valued results to binary for scoring
            temp = []
            for y in self.y_pred:
                if y > 0.0:
                    temp.append(1.0)
                else:
                    temp.append(-1.0)
            self.y_pred = temp

        return metrics.accuracy_score(self.y_test, self.y_pred)


    def getReport(self,save = 1, get_top_words = 0):       # returns report
        report = ""
        if get_top_words == 1:
            if hasattr(self.mlModel, 'coef_'):
                    report += "Dimensionality: " + str(self.mlModel.coef_.shape[1])
                    report += "\nDensity: " +  str(density(self.mlModel.coef_))

                    rank = np.argsort(self.mlModel.coef_[0])
                    top10 = rank[-20:]
                    bottom10 = rank[:20]
                    report += "\n\nTop 10 keywords: "
                    report += "\nPositive: " + (" ".join(self.feature_names[top10]))
                    report += "\nNegative: " + (" ".join(self.feature_names[bottom10]))

        score = metrics.accuracy_score(self.y_test, self.y_pred)
        report += "\n\nAccuracy: " + str(score)
        report += "\nClassification report: "
        report += "\n\n" + str(metrics.classification_report(self.y_test, self.y_pred,target_names=["Negative","Positive"]))
        report += "\nConfusion matrix: "
        report += "\n\n" + str(metrics.confusion_matrix(self.y_test, self.y_pred)) + "\n\n"

        if save == 1:
            with open(self.model_path + "report.txt", "w") as text_file:
                text_file.write(report)

        return report


    def crossVal(self,folds = 5, dim_red = 50,full_iter = 0, save = 1):        # returns report # Caution: resets train and test X,y
        skf = cross_validation.StratifiedKFold(self.y_train, n_folds = folds,shuffle=True)
        print(skf)
        master_report = ""

        X_copy = self.X_train
        y_copy = self.y_train

        for train_index, test_index in skf:
            self.X_train, self.X_test = X_copy[train_index], X_copy[test_index]
            self.y_train, self.y_test = y_copy[train_index], y_copy[test_index]
            self.featurizeXY(0)
            self.reduceDimension(0,dim_red)
            self.trainModel()
            self.testModel()
            master_report += self.getReport(save = 0,get_top_words = 0)
            if full_iter == 1:
                continue
            else:
                break

        if save == 1:
            with open(self.model_path + "master_report.txt", "w") as text_file:
                text_file.write(master_report)

        return master_report


    def save_obj(self,obj, name ):
        with open(self.model_path + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f,  protocol=2)


    def saveModel(self):        # saves in model path
        self.save_obj(self.mlModel, self.name + "_model")
        self.save_obj(self.vectorizer, self.name + "_vectorizer")
        self.save_obj(self.chi2, self.name + "_feature_selector")


    def plot(self):
        '''
        beta (Just plotting the model) (Not working)
        '''

        h = .02  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        Z = self.mlModel.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(self.name)
        plt.savefig(self.model_path + 'plot.png')
