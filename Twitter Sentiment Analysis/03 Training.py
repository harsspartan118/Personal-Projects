# Model building-

# PyPI: textblob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

trainset = pd.read_csv('trainsetCleaned.csv', encoding='utf-8')
testset = pd.read_csv('testsetCleaned.csv', encoding='utf-8')

# Dropping the index column read from file.
# trainset.drop('Unnamed: 0', axis=1, inplace=True)
# testset.drop('Unnamed: 0', axis=1, inplace=True)

trainset.info()
#	Only trainset contains null entries.

trainset.dropna(inplace=True)
trainset.index
#	Index is not contigous. 1599993 missing.

trainset.reset_index(inplace=True)

xTrain, xTest, yTrain, yTest = train_test_split(trainset.text, trainset.polarity, test_size=0.02)
xValidation, xTest, yValidation, yTest = train_test_split(xTest, yTest, test_size=0.5)
#	Split into 98-1-1 percent.

xTrain.size
#	Approx 16000 in validation and test.
# No need for multi fold validation scoring since dataset is large.

xTest[yTest == 0].size
xTest[yTest == 4].size
#	Equally distributed positive and negative points.

# Using Zero rule clasifier an TextBlob as baselines.
from textblob import TextBlob

tbPolarities = [TextBlob(text).sentiment.polarity for text in xValidation]
#	Polarities range from -1 to 1
tbPredictions = [4 if polarity > 0 else 0 for polarity in tbPolarities]

accuracy_score(yValidation, tbPredictions)
#	62.4 %
confusionMat = confusion_matrix(yValidation, tbPredictions, labels=[0, 4])
confusionDF = pd.DataFrame(confusionMat,
							columns=['Negative', 'Positive'],
							index=['Predicted Negative', 'Predicted Positive'])
#	TN, FN
#	FP, TP
print(classification_report(yValidation, tbPredictions, labels=[0, 4]))
#	Average F1- 62%

# Feature engineering-
# Consider how many features to keep in the bag of words.
# Consider Tfidf vectorizer or Count vectorizer.		(Tfidf performs better here)
# Consider whether to keep stop words.		(Keeping performs better here)
# Consider N-grams.		(Upto trigram performs better here)

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

models = []

# Heavy memory load operations. Do not multithread. Upto 5.6 GB memory.
def getAccuracyPerNumFeatures() :
	numFeatureRange = range(10000, 100001, 10000)
	accuracyResults = []
	
	for numFeatures in numFeatureRange :
		logReg = LogisticRegression()
		#	No need to change regularization parameter. Change number of features instead.
		tfidfVec = TfidfVectorizer(max_features=numFeatures,
								stop_words=None,
								ngram_range=(1, 3))
		pl = make_pipeline(tfidfVec, logReg)
	
		print("Calculating accuracy with " + str(numFeatures) + " features...")
		pl.fit(xTrain, yTrain)
		yPredicted = pl.predict(xValidation)
		accuracy = accuracy_score(yValidation, yPredicted)
		accuracyResults.append((numFeatures, accuracy))
		models.append((numFeatures, tfidfVec, logReg))
		print("Accuracy = " + str(accuracy))
		
	return accuracyResults

accuracyResults = getAccuracyPerNumFeatures()
accuracyResults.max()
#	Best at 100000 features.

import csv
writer = csv.writer(open('AccuracyPerNumFeatures.csv', 'w', newline=""))
writer.writerows(accuracyResults)
del(writer)

accuracyResults = pd.DataFrame(accuracyResults, columns=['numFeatures', 'accuracy'])
plt.plot(accuracyResults.numFeatures, accuracyResults.accuracy, color='royalblue')
plt.show()

# Train final model.
# May consider creating an ensamble of multiple models. Performs worse in this case.
finalModel = models[-1][1:]
#	Save the vectorizer and model.

#logReg = LogisticRegression()
#tfidfVec = TfidfVectorizer(max_features=100000,
#						stop_words=None,
#						ngram_range=(1, 3))
#termDocMatrix = tfidfVec.fit_transform(xTrain)
#logReg.fit(termDocMatrix, yTrain)

predictions = finalModel[1].predict(finalModel[0].transform(xTest))
accuracy_score(predictions, yTest)
# 82.5%
