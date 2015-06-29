# Author : Satish Palaniappan
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
