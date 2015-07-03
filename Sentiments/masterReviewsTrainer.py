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

def chkMake(dirPath):
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)
	return dirPath

def getFilesDir(pathDir):
	files = [f for f in os.listdir(pathDir) if ".pkl" in f]
	fd = os.listdir(pathDir)
	directories = list(set(fd)-set(files))
	return[files,directories]

baseModelPath = "../model/review/"
baseDataPath = "../data/pickle/review/"


files,directories = getFilesDir(baseDataPath)

topDomains = directories + files	# All top level domains
# topDomains = ["places.pkl"]

subDomainMap = dict()

for di in directories:
	subDomainMap[di] = os.listdir(baseDataPath + di + "/")

# tempPaths = []

for domain in topDomains:
	data_path = []
	if ".pkl" in domain:
		data_path.append (baseDataPath + domain)
	else:
		for i in subDomainMap[domain]:
			data_path.append (baseDataPath + domain + "/" + i)

	for p in data_path:
		print(p)
		# newLearner = learner.TextLearner(p)
		with learner.TextLearner(p) as newLearner:
			data_type = newLearner.load_data()

			if len(data_path) == 1:
				model_path = baseModelPath + domain.replace(".pkl","") + "/"
			else:
				model_path = baseModelPath + domain + "/" + p.replace(baseDataPath + domain + "/","").replace(".pkl","") + "/"

			if data_type == 2 or data_type == 3:
				chkMake(model_path)
				newLearner.addModelDetails(model_p = model_path)
				learn_save(newLearner,0,1)
				# tempPaths.append(model_path)
			else:
				for j in ["title","review_text"]:
					chkMake(model_path + j + "/")
					newLearner.addModelDetails(model_p = (model_path + j + "/"))
					if j == "title":
						learn_save(newLearner,3,1)
						# tempPaths.append((model_path + j + "/"))
					else:
						learn_save(newLearner,0,1)
						# tempPaths.append((model_path + j + "/"))

# for t in tempPaths:
# 	print(t)
