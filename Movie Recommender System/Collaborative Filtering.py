import numpy as np
import pandas as pd

data = pd.read_csv('G:\\Other Stuff\\Big data\\My Projects\\Movie Recommender System\\movielens-dataset-2016-small\\ratings.csv')
data.drop('timestamp', axis=1, inplace=True)

movieTitles = pd.read_csv('G:\\Other Stuff\\Big data\\My Projects\\Movie Recommender System\\movielens-dataset-2016-small\\movies.csv')

#   Confirm that user id and movie ids are contigous.
uuids = data.userId.unique()
umids = data.movieId.unique()

uuids.sort()
np.all(np.diff(uuids) == 1)
#   True
umids.sort()
np.all(np.diff(umids) == 1)
#   False

#   Get number of users and movies
len(uuids)
#   671
len(umids)
#   9066

#   Convert to a matrix form
umids = pd.Series(range(1, len(umids)+1), index=umids)
ratings = np.zeros((len(uuids), len(umids)))

#   Thousands of times faster to use numpy. Since written in C and avoids GIL.
npdata = data.values
ratings[npdata[..., 0].astype('int') - 1, umids[npdata[..., 1].astype('int')] - 1] = npdata[..., 2]

#   Calculate cosine similarity amongst all users and amongst all movies.
#   Heavy operation. Can limit the number of items per category for calculation.
#   User interests may differ for different categories, may cluster the items first.
from sklearn.metrics.pairwise import cosine_similarity
userSim = cosine_similarity(ratings)

#   We have fewer users than items, less expensive to predict based on user-user similarity.
# itemDist = cosine_similarity(ratings.T)

#   May cluster users before using for prediction. Or predict based on a subset of users.
#   For user based filtering, must consider user bias into account.
#   Final predictions are a similarity weighted average of rating deviations added to user bias.

#   Avoid unrated values from the mean.
#   Cannot directly filter as it will flatten the array. Must calculate manually per column.
#   Calcutate deviations from ratings per user to eliminate user bias.
def getMeanAndDeviationPerUser(ratings) :
    means = np.zeros(ratings.shape[0])
    deviations = np.zeros(ratings.shape)
    for i in range(ratings.shape[0]) :
        filteredRow = ratings[i, ratings[i, ...] != 0]
        means[i] = filteredRow.mean()
        
        for j in range(ratings.shape[1]) :
            if ratings[i, j] != 0 :
                deviations[i, j] = ratings[i, j] - means[i]
    return means, deviations

meanRatings, ratingDeviations = getMeanAndDeviationPerUser(ratings)

def predictRatingsForUser(userIndex, meanRatings=meanRatings, ratingDeviations=ratingDeviations) :
    pred = meanRatings[userIndex] + (userSim[userIndex, ...].dot(ratingDeviations)
                                        / userSim[userIndex, ...].sum())
    return pred

def getRecommendations(userId) :
    pred = predictRatingsForUser(userId, meanRatings, ratingDeviations)
    predSer = pd.Series(pred, index=umids)
    
    ratingSer = pd.Series(ratings[userId], index=umids)
    ratingSer = ratingSer[ratingSer != 0].sort_values(ascending=False)
    ratedTitles = movieTitles.assign(rating=ratingSer).loc[ratingSer.index][['title', 'rating']]
    
    for i in predSer.index :
        if i in ratingSer.index :
            predSer.drop(i, inplace=True)
    recom = predSer.sort_values(ascending=False)[0:10]
    recomTitles = movieTitles.loc[recom.index, 'title']

    return ratedTitles, recomTitles

ratedTitles, recomTitles = getRecommendations(12)
ratedTitles
recomTitles

#	Can get errors by comparing predicted ratings to existing ratings.