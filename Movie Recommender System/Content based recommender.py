#	Collecting movie features-
import numpy as np
import pandas as pd

moviedata = pd.read_csv('movielens-dataset-2016-small\\movies.csv')
crewdata = pd.read_csv('Intermediate results\\crew-details.csv')

# Join data based on movie id.
data=moviedata.join(crewdata, on='movieId', lsuffix='movie', rsuffix='crew')
data.drop(['movieIdcrew', 'imdbId'], axis=1, inplace=True)
data.rename(columns={'movieIdmovie':'movieId'}, inplace=True)
# Not including tag data as its too sparse.

moviedata.genres.notna().value_counts()
#	No null values
#	Director has 3472 nulls
#	Writer has 3472 nulls
#	Actor has 1514 nulls
#	Actress has 1514 nulls
#	Composer has 1514 nulls

# Convert genre to list.
data.genres = data.genres.apply(str.split, args=('|', ))

#	Saving to csv will convert lists to strings.
#	Use HDF5 (Hierarchical data format) for fast access with large datasets.
#	Requires PyTables (PYPI- tables)
import tables
store = pd.HDFStore('movieFeatures-data.h5')
store['data'] = data

#	To load-
#	store = pd.HDFStore('movieFeatures-data.h5')
#	data = store['data']

# 	Numeric vs Ordinal vs Nominal features-
#		Ordinal features represent categories that can be compared. E.g. low, mid, high
#		Nominal features represent categories that cannot be compared. E.g. apple, orange
#	To handle ordinal features, transform them into columns using preprocessing.OrdinalEncoder
#	For nominal, use OneHotEncoder
#	Can use LabelEncoder to convert labels to ints. Doesn't work here with multi class elements.
#	MultiLabelBinarizer works here on a column by column basis.

def getLabels(col) :
	res = set()
	for items in col :
		if type(items) == list :
			res.update(items)
	return res

labels = getLabels(data.genres)
#	Found label- (no genres listed). Must treat as NaN.

# Find empty genre indexes-
def getGenreEmpty() :
	res = []
	for i in data.index :
		if '(no genres listed)' in data.at[i, 'genres'] :
			res.append(i)
	return res
	
indexes = getGenreEmpty()
#	Found 18

# Drop rows if all features are empty
def dropIfEmpty(indexes) :
	for i in indexes :
		if (np.isnan(data.directorIds[i])
				and np.isnan(data.writerIds[i])
				and np.isnan(data.actorIds[i])
				and np.isnan(data.actressIds[i])
				and np.isnan(data.composerIds[i])) :
			data.drop(i, axis=0, inplace=True)
			print("Dropping " + str(i))

dropIfEmpty(indexes)
#	All 18 dropped.

len(getLabels(data.writerIds))
#	19 genres
#	2917 directorIds
#	7867 writerIds		--	Too many for 9000 movies
#	2203 actorIds
#	2203 actressIds
#	2203 composerIds

data.drop('writerIds', axis=1, inplace=True)

import scipy.sparse as ssp
from sklearn.preprocessing import MultiLabelBinarizer

# Don't need title, movie id to be binarized.
features = data.set_index('movieId')[
						['genres', 'directorIds', 'actorIds',
						 'actressIds', 'composerIds']]

# Replce all NaN with empty lists to denote no ones.
def replaceNans(row) :
	for col in row.index :
		if type(row[col]) is not list :
			row[col] = []
	return row
	
features.apply(replaceNans, axis=1)

def binarizeData(data) :
	mlbin = MultiLabelBinarizer(sparse_output=True)
	labels = []
	csrMat = ssp.csr_matrix((len(data), 0))
	for col in data.columns :
		csrMat = ssp.hstack([csrMat, mlbin.fit_transform(data[col])])
		labels.append(mlbin.classes_)
	return csrMat, labels
	
binData, labels = binarizeData(features)
#	More features than examples. Will overfit.
#	No need for dimentionality reduction as recommender systems only work their own data.

import pickle as pkl

file = open('movieVectorMatrix.pkl', 'wb')
pkl.dump(binData, file)
file.close()

file = open('movieVectorLabels.pkl', 'wb')
pkl.dump(labels, file)
file.close()

#	To calculate similarity between movies, can use cosine, euclidian or pearson similarity.
#	Cosine similarity-
#		cos(theta) = dot product of vectors / product of vector magnidudes
#		It measures the tendency of two vectors to represent the same thing (angle)
#		It doesn't account for the intensity with which a vector represents something (magnitude)
#		It maintains its value if a subset of the data is used.
#	Euclidean similarity-
#		It consideres the direction and magnitude of data points.
#		It is suseptible to clustering together low magnitude points with different directions.
#	Pearson similarity-
#		It is similar to cosine similarity, but it works on mean normalized values.
#		It is suseptible to change if subset of the data is used.
from sklearn.metrics.pairwise import cosine_similarity
simMat = cosine_similarity(binData, dense_output=False)

simMat[0:4, 0:4].todense()

#	Make recommendations by comparing similarity.
data.reset_index(inplace=True)

#	Use title as index and position as column.
movieTitles = pd.Series(data.index, index=data.title)

def getRecommendations(title, simMat) :
	# Find movie index
	movieIndex = movieTitles[title]
	# Get similarity scores against this movie. Matrix gives multi-dim array.
	simScores = simMat[movieIndex].toarray()[0]
	# Sort the scores. Reverse with negative step.
	order = simScores.argsort()[ : :-1]
	# Show 10 best scoring movies.
	recom = order[1:11]
	
	return data.title[recom]
	
#	Evaluation is best done against human intuition. By collecting feedback from people.