# Author : Satish Palaniappan
__author__ = "Satish Palaniappan"

import os
import pickle
from os import path
import learner

'''
Pickle Formats
2#review,pos_neg
3#review,pos_neg,score
4#review,pos_neg,score,title
'''

def learn_save(newLearner,feature_i,label_i):
	newLearner.clearOld()
	newLearner.loadXY(feature_index = feature_i,label_index = label_i)
	print("Vecs Loaded!")
	newLearner.featurizeXY()
	print("Features extracted!")
	newLearner.reduceDimension()
	newLearner.trainModel()
	print("Model Trained!")
	newLearner.saveModel()
	print("Model Saved!")

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

baseModelPath = "../model/microblogs/"
baseDataPath = "../data/pickle/microblogs/tweets.pkl"

with learner.TextLearner(baseDataPath) as newLearner:
	newLearner.load_data()
	newLearner.addModelDetails(model_p = baseModelPath)
	learn_save(newLearner,0,1)
