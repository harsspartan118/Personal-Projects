# Data visualization and analysis-

# PyPI: wordcloud, seaborn, bokeh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

import wordcloud as wc
negativeTweets = trainset.text[trainset.polarity == 0]
negativeString = ' '.join([s for s in negativeTweets])

positiveTweets = trainset.text[trainset.polarity == 4]
positiveString = ' '.join([s for s in positiveTweets])

negWordCloud = (wc.WordCloud(width=1600, height=800, max_font_size=200)
									.generate(negativeString))
posWordCloud = (wc.WordCloud(width=1600, height=800, max_font_size=200,
										colormap='magma')
									.generate(positiveString))
									
plt.figure(1)
plt.imshow(negWordCloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative words')

plt.figure(2)
plt.imshow(posWordCloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive words')

plt.show()

import nltk
# from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords

# Remove stop words before frequency plotting.
# tokenizer = WordPunctTokenizer()
nltk.download('stopwords')
stopWordSet = set(stopwords.words('english'))

testsetTokenized = pd.DataFrame(index=range(len(testset)), columns=['polarity', 'tokens'])
trainsetTokenized = pd.DataFrame(index=range(len(trainset)), columns=['polarity', 'tokens'])

def tokenizeTestset() :
	for i in range(len(testset)) :
		testsetTokenized.iloc[i]['polarity'] = testset.iloc[i]['polarity']
		testsetTokenized.iloc[i]['tokens'] = [token
			for token in """tokenizer.tokenize(testset.iloc[i]['text'])""" testset.iloc[i]['text'].split(' ')
				if token not in stopWordSet]

def tokenizeTrainsetSlice(lowerIndex, upperIndex) :
	for i in range(lowerIndex, upperIndex) :
		trainsetTokenized.iloc[i]['polarity'] = trainset.iloc[i]['polarity']
		trainsetTokenized.iloc[i]['tokens'] = [token
			for token in """tokenizer.tokenize(trainset.iloc[i]['text'])""" trainset.iloc[i]['text'].split(' ')
				if token not in stopWordSet]

		if (i % 50000 == 0) :
			print("Record #" + str(i) + " processed.")

import threading
# Can use GPU processing with CUDA. Directly supported in Python.
#	GPU vs CPU processing-
#		1. GPU cores are better at data intensive tasks compared to handling different tasks.
#			I.e. Several cores doing the same task for different parts of data.
#		2. GPUs are limited by memory transfer speed.
#			I.e. Prefer tasks that are computationally expensive vs read-write expensive.
#			Generally 500x instructions per read-write.
#		3. GPUs don't work well with variable length data points. E.g. strings
#			Since strings don't involve heavy computations but heavy data transfer.
#			GPUs cannot assign tasks dynamically to their cores. Clusters do a single task.

print("Tokenizing datasets...")
sliceLen = len(trainset) // 4
#	399092
thread1 = threading.Thread(target=tokenizeTrainsetSlice, args=(0, sliceLen))
thread2 = threading.Thread(target=tokenizeTrainsetSlice, args=(sliceLen, sliceLen*2))
thread3 = threading.Thread(target=tokenizeTrainsetSlice, args=(sliceLen*2, sliceLen*3))
thread4 = threading.Thread(target=tokenizeTrainsetSlice, args=(sliceLen*3, len(trainset)))

thread1.start()
thread2.start()
thread3.start()
thread4.start()

tokenizeTestset()

thread1.join()
thread2.join()
thread3.join()
thread4.join()

print("Datasets tokenized.")

trainsetTokenized.to_csv('trainsetTokenized.csv', encoding='utf-8', index=False)
testsetTokenized.to_csv('testsetTokenized.csv', encoding='utf-8', index=False)	

# trainset = trainsetTokenized
from sklearn.feature_extraction.text import CountVectorizer

cvecTrain = CountVectorizer(max_features=10000)
cvecTrain.fit(trainset.tokens)
negMatrix = cvecTrain.transform(trainset[trainset.polarity == 0].tokens)
posMatrix = cvecTrain.transform(trainset[trainset.polarity == 4].tokens)
negTermFreq = np.squeeze(np.asarray(np.sum(negMatrix, axis=0)))
posTermFreq = np.squeeze(np.asarray(np.sum(posMatrix, axis=0)))
termFreqData = pd.DataFrame([negTermFreq, posTermFreq],
								columns=cvecTrain.get_feature_names(),
								index=('NegFreq', 'PosFreq')).transpose()
#	10000 x 2 dataframe. Would have generated 400000+ features otherwise.
termFreqData.to_csv('trainTermFreq.csv')

# Plot as bar chart the 50 most frequent tokens per class.
dataNegFreqSorted = termFreqData.sort_values(by='NegFreq', ascending=False)[:50]
dataPosFreqSorted = termFreqData.sort_values(by='PosFreq', ascending=False)[:50]

barPositions = np.arange(50)
plt.subplot(2, 1, 1)
plt.bar(barPositions, dataNegFreqSorted.NegFreq, alpha=0.5, color='#C51162')
plt.xticks(barPositions, dataNegFreqSorted.index, rotation='vertical')
plt.ylabel('Frequency')
plt.title('50 most frequent negative tokens')

plt.subplot(2, 1, 2)
plt.bar(barPositions, dataPosFreqSorted.PosFreq, alpha=0.5, color='#018786')
plt.xticks(barPositions, dataPosFreqSorted.index, rotation='vertical')
plt.ylabel('Frequency')
plt.title('50 most frequent positive tokens')
plt.show()

# Scatter plot of tokens.
import seaborn as sb
sb.regplot(x='NegFreq', y='PosFreq', data=termFreqData,
						fit_reg=False, scatter_kws={'alpha' : 0.5})
plt.xlabel('Negative token frequency')
plt.ylabel('Positive token frequency')
plt.title('Negative vs Positive token frequencies')
plt.show()

# To find more informative metrics-
# Words may have very high frequency in both classes, or may have too small total frequency.
# Ranking by frequency cannot relate word to entire corpus.
# Use positive rate = positive frequency / total frequency
# Use frequency percent = positive frequency / sum(positive frequencies of all words)
# Both combined should specify where a word stands in the corpus.
# frequency percent values much smaller than positive rate values.
# Arithmetic and harmonic means too skewed towards larger and smaller values.
# Use Cumulative distribution function instead. Similar to percentile score.
# Harmonic mean of CDFs of above metrics provides informative words.

# data = pd.read_csv('trainTermFreq.csv', index_col=0)
import scipy.stats as sps
# sps.norm- Class for normally distributed random variable methods.

data['PosRate'] = data.PosFreq / (data.PosFreq + data.NegFreq)
data['NegRate'] = data.NegFreq / (data.PosFreq + data.NegFreq)

totalPosFreq, totalNegFreq = data.PosFreq.sum(), data.NegFreq.sum()
data['PosFreqPercent'] = data.PosFreq / totalPosFreq
data['NegFreqPercent'] = data.NegFreq / totalNegFreq

posRateMean, negRateMean = data.PosRate.mean(), data.NegRate.mean()
posRateStd, negRateStd = data.PosRate.std(), data.NegRate.std()
data['PosRateCDF'] = sps.norm.cdf(data.PosRate, posRateMean, posRateStd)
data['NegRateCDF'] = sps.norm.cdf(data.NegRate, negRateMean, negRateStd)

posPercentMean = data.PosFreqPercent.mean()
negPercentMean = data.NegFreqPercent.mean()
posPercentStd = data.PosFreqPercent.std()
negPercentStd = data.NegFreqPercent.std()
data['PosPercentCDF'] = sps.norm.cdf(data.PosFreqPercent, posPercentMean, posPercentStd)
data['NegPercentCDF'] = sps.norm.cdf(data.NegFreqPercent, negPercentMean, negPercentStd)

data['PosCdfMean'] = sps.hmean([data.PosRateCDF, data.PosPercentCDF])
data['NegCdfMean'] = sps.hmean([data.NegRateCDF, data.NegPercentCDF])
data.to_csv('trainTermStats.csv')

data.sort_values(by='NegCdfMean', ascending=False)
#	Commonly used descriptive words at top.

# Scatter plot of tokens by CDF means.
sb.regplot(x='NegCdfMean', y='PosCdfMean', data=data,
						fit_reg=False, scatter_kws={'alpha' : 0.5})
plt.xlabel('Negative token CDF mean')
plt.ylabel('Positive token CDF mean')
plt.title('Negative vs Positive token by CDF means')
plt.show()

# Interactive Bokeh plot-
import bokeh.plotting as bplt
import bokeh.io as bio
import bokeh.models as bmods
import bokeh.palettes as bpals

bio.output_file('Neg vs Pos CDF - Bokeh plot.html')
fig = bplt.figure(x_axis_label='NegCdfMean', y_axis_label='PosCdfMean',
				plot_height=1000, plot_width=1000)
colorMapper = bmods.LinearColorMapper(palette=bpals.RdYlGn11, 
								low=data.NegCdfMean.min(),
								high=data.NegCdfMean.max())
fig.circle('NegCdfMean', 'PosCdfMean', size=5, alpha=0.3, source=data,
			color={'field':'NegCdfMean', 'transform':colorMapper})
hoverTool = bmods.HoverTool(tooltips=[('Token', '@index')])
fig.add_tools(hoverTool)
bio.show(fig)
bplt.reset_output()