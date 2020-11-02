# Preparing the data-

# Dataset-
# Sentiment140 from Stanford (http://help.sentiment140.com/for-students/)
# The data is a CSV with emoticons removed. Data file format has 6 fields:
# 0 - the polarity of the tweet (0=negative, 2=neutral, 4=positive)
# 1 - the id of the tweet (2087)
# 2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
# 3 - the query (lyx). If there is no query, then this value is NO_QUERY.
# 4 - the user that tweeted (robotickilldozr)
# 5 - the text of the tweet (Lyx is cool)

# PyPI: beautifulsoup4, lxml, flashtext, nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labels = ['polarity', 'id', 'date', 'query', 'user', 'text']
trainset = pd.read_csv('trainingandtestdata\\trainset.csv', names=labels, encoding='latin-1')
testset = pd.read_csv('trainingandtestdata\\testset.csv', names=labels, encoding='latin-1')
#	UTF-8 is variable length encoding (1 - 4 bytes per char).
#	Encodes 1 million + unicode code points.
#	Latin-1 is fixed length (1 byte). Encodes 256 code points.
#	Both are compatible with ASCII from 0 to 127. UTF-8 jumps to 2 byte from 128.

trainset.index
#	1.6 million in training, 500 in test.
trainset.columns
trainset.head()

trainset.polarity.value_counts()
#	Equally distributed in positive and negative classes in training.
#	Distributed along all three classes in test.

trainset.drop(['id', 'date', 'query', 'user'], axis=1, inplace=True)
testset.drop(['id', 'date', 'query', 'user'], axis=1, inplace=True)

trainset[trainset.polarity == 0].head()
trainset[trainset.polarity == 0].index
#	All negative examples in training from 0-799999. All positive from 800000+.
#	All examples spread around in test.

trainTextLen = np.array([len(t) for t in trainset.text])
testTextLen = np.array([len(t) for t in testset.text])
#	Space efficient way-
#	Use a generator. But ndarrays need to know length in advance.
#	trainlen = np.empty(len(trainset.text))
#	for i, x in enumerate((len(t) for t in trainset.text)): trainlen[i] = x

np.max(trainTextLen)
# Max length goes above twitter's 140 character limit.

#	To represent data distribution (central tendency and spread)-
#	Mean and std deviation-
#		Work in percentage terms. About 68% points lie within 1 std dev, 95 lie in 2 std dev.
#		Only effective for normal distribution.
#		Don't handle outliers well. I.e. different distributions may have same mean and std dev.
#	Median and Interquartile range (IQR)-
#		Work in percentile terms.
#		Work well on non-normal distributions and represent outliers well.
#	IQR-
#		Distribution divided into 4 quartiles. Center split at median.
#		Splits at 25 and 75 percentile (midway from median to lowest / highest).
#		Centeral 50% is difference of 75 and 25 percentile. Called IQR.
#	Box - Whisker plots-
#		Represent centeral 50% as box with a median line.
#		Whiskers repsent outer quartiles (min to max).
#		Whiskers may represent points within 1.5 IQR from inner quartiles. (Turkey boxplot)
plt.subplot(1, 2, 1)
plt.boxplot(trainTextLen)	# Turkey by default. Set whis='range' for min / max variant.
plt.subplot(1, 2, 2)
plt.boxplot(testTextLen)
plt.show()

trainset.text[trainTextLen > 140].head()
# HTML codes included in data.
# @mentions, URLs useless for sentiment.
# UTF-8 Byte Order Marks (utf-8-sig) used to specify string as UTF-8.

#	BeautifulSoup4-	HTML and XML parsing library.
#		Slow by default. Was best for handling poorly formed data.
#	lxml-	HTML and XML parsing library.
#		Fast. Was not good for poorly formed data.
#	Both libraries now support using each other for best results.
from bs4 import BeautifulSoup
BeautifulSoup(trainset.text[400], 'lxml').get_text()

# [] - group of symbols. Use ^ to get complement of the group.
# () - class of fixed strings.
# ^, $ - start and end of string.
# *, +, ? - 0 or more, 1 or more, 0 or 1 of preceding RE.
# . - any symbol except newline.
import re
refPattern = '@[a-zA-Z0-9_]+'
re.sub(refPattern, '', trainset.text[0])

httpPattern = 'https?://[^ ]+'
re.sub(httpPattern, '', trainset.text[0])

wwwPattern = 'www.[^ ]+'
re.sub(wwwPattern, '', trainset.text[50])

hexPattern = '\\\\x[a-f0-9\\\\x]+'
re.sub(hexPattern, '?', str(trainset.text[226].encode('utf-8'))[2 : -1])
# Need \\\\ to represent \\ after going through python interpreter.
# Slicing to remove quots and byte prefix.

alnumPattern = '[^A-Za-z0-9 ]'
re.sub(alnumPattern, '', trainset.text[193])

negativeWordDict = {
	"is not" : ["isn't", "ain't"],			"are not" : ["aren't"],
	"was not" : ["wasn't"],				"were not" : ["weren't"],
	"have not" : ["haven't"],			"has not" : ["hasn't"],
	"had not" : ["hadn't"],				"will not" : ["won't"],
	"would not" : ["wouldn't"],		"do not" : ["don't"],
	"does not" : ["doesn't"],			"did not" : ["didn't"],
	"can not" : ["can't"],					"could not" : ["couldn't"],
	"should not" : ["shouldn't"],	"might not" : ["mightn't"],
	"must not" : ["mustn't"]
}

import flashtext as ft
# Fast find and replace words in text. Uses Trie data structure.
keywordProcessor = ft.KeywordProcessor()
keywordProcessor.add_keywords_from_dict(negativeWordDict)

from nltk.tokenize import WordPunctTokenizer
# Generates separate tokens for words and punctuations in a list.
tokenizer = WordPunctTokenizer()

def cleanText(text) :
	cleanedText = BeautifulSoup(text, 'lxml').get_text()
	cleanedText = re.sub(hexPattern, '?', str(cleanedText.encode('utf-8'))[2 : -1])
	
	combinedPattern = '|'.join([httpPattern, refPattern])
	cleanedText = re.sub(combinedPattern, '', cleanedText)
	cleanedText = re.sub(wwwPattern, '', cleanedText)
	
	cleanedText = cleanedText.lower()
	cleanedText = keywordProcessor.replace_keywords(cleanedText)
	
	cleanedText = re.sub(alnumPattern, '', cleanedText)
	words = tokenizer.tokenize(cleanedText)
	return ' '.join(words)

trainsetCleaned = pd.DataFrame(index=range(len(trainset)), columns=['polarity', 'text'])
testsetCleaned = pd.DataFrame(index=range(len(testset)), columns=['polarity', 'text'])
	
def cleanTrainsetSlice(lowerIndex, upperIndex) :
	for i in range(lowerIndex, upperIndex) :
		trainsetCleaned.iloc[i]['polarity'] = trainset.iloc[i]['polarity']
		trainsetCleaned.iloc[i]['text'] = cleanText(trainset.iloc[i]['text'])
		
		if (i % 50000 == 0) :
			print("Record #" + str(i) + " processed.")

def cleanTestset() :
	for i in range(len(testset)) :
		testsetCleaned.iloc[i]['polarity'] = testset.iloc[i]['polarity']
		testsetCleaned.iloc[i]['text'] = cleanText(testset.iloc[i]['text'])
			
import threading

print("Cleaning datasets...")
thread1 = threading.Thread(target=cleanTrainsetSlice, args=(0, 400000))
thread2 = threading.Thread(target=cleanTrainsetSlice, args=(400000, 800000))
thread3 = threading.Thread(target=cleanTrainsetSlice, args=(800000, 1200000))
thread4 = threading.Thread(target=cleanTrainsetSlice, args=(1200000, 1600000))

thread1.start()
thread2.start()
thread3.start()
thread4.start()

cleanTestset()

thread1.join()
thread2.join()
thread3.join()
thread4.join()

print("Datasets cleaned.")

trainsetCleaned.to_csv('trainsetCleaned.csv', encoding='utf-8', index=False)
testsetCleaned.to_csv('testsetCleaned.csv', encoding='utf-8', index=False)
# Index is false to prevent writing index to file.