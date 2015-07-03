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


import learner

# # print(str(max(self.y_pred)) + "" + str(min(self.y_pred)))
# # temp = list(self.y_pred)
# # temp = sorted(temp)
# # print(str(sum(temp[:100])/float(len(temp[:100]))) + str(sum(temp[-100:])/float(len(temp[-100:]))))

# Scoring Function
maxmin=1.5
buckets = [maxmin - i*(maxmin*2/11) for i in range(0,12)]
def getScore (x):
	for i in range(0,11):
		if x > buckets[0]:
			return 5
		if x < buckets[len(buckets)-1]:
			return -5
		if buckets[i] >= x >= buckets[i+1]:
			return(5-i)

print(getScore(-0.2335241799033111))

# newLearner = learner.TextLearner("../../data/pickle/general/general.pkl","../../model/testing/","")

# newLearner.load_data()
# newLearner.loadXY(feature_index = 0,label_index = 1)
# # newLearner.addDataPath("../data/pickle/electronics_tech/general/pros_cons")
# # newLearner.load_data()
# # newLearner.loadXY(feature_index = 0,label_index = 1)
# newLearner.crossVal(folds=2,full_iter=0)
# # print(max(newLearner.y_pred) + "" + min(newLearner.y_pred))
# # newLearner.saveModel()
# # newLearner.plot()

# newLearner.data_path = "../../data/pickle/comments/comments.pkl"
# newLearner.load_data(TrTe = 1)
# newLearner.loadXY(TrTe = 1,feature_index = 0,label_index = 1)
# newLearner.featurizeXY(0)
# newLearner.reduceDimension(0,50)
# newLearner.trainModel()
# newLearner.testModel()
# print(max(newLearner.y_pred) + "" + min(newLearner.y_pred))
# newLearner.getReport()
