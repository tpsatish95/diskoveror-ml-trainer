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
