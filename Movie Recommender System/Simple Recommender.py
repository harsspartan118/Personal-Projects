#	Calculate weighter rating based on averate rating and number of votes.
#	Problems with simple metrics-
#		Mean rating will favour new movies with very few votes.
#		Total rating will favour old movies with lots of poor votes.

# Formula for weighted rating-
#		(v / (v+m)) * R + (m / (v+m)) * C
#
#		v is the number of votes received by the movie.
#		m is the minimum votes required to be listed. Using 90 percentile score.
#		R is the average rating given to the movie.
#		C is the average rating received by all movies in the dataset. Average R.

# Combine required dataset

ratingdata=pd.read_csv('ratings.csv')
moviedata=pd.read_csv('movies.csv')

ratingdata.head(50)
#	Sorted by user id then movie id.
#	Ratings between 1-5

moviedata['totalRating'] = 0.0
moviedata['numVotes'] = 0

#	Use movie id as index temporarly.
moviedata.reset_index(inplace=True)	# Save the index as a column.
moviedata.set_index('movieId', inplace=True)

def collectRatings() :
	for i in ratingdata.index :
		movieId = ratingdata.at[i, 'movieId']
		moviedata.at[movieId, 'totalRating'] += ratingdata.at[i, 'rating']
		moviedata.at[movieId, 'numVotes'] += 1
		
collectRatings()

# Assign original index.
moviedata.reset_index(inplace=True)
moviedata.set_index('index', inplace=True)

#	Calculate mean ratings.
moviedata.totalRating = moviedata.totalRating / moviedata.numVotes
moviedata.rename(columns={'totalRating' : 'meanRating'}, inplace=True)

#	Take m as the 90 percentile value.
#	I.e. votes received by a movie that has more votes than 90 percent movies in the dataset.
minReqVotes = moviedata.numVotes.quantile(0.9)
moviedata.to_csv('rated_movies.csv')

#	Filter out movies with minimum required votes.
moviedata = moviedata.loc[moviedata.numVotes > minReqVotes]
moviedata.shape
#		883 movies left.

#	C will be average of average ratings.
C = moviedata.meanRating.mean()

# Calculate weighted rating
moviedata['score'] = 0.0
m = minReqVotes
v = moviedata.numVotes
R = moviedata.meanRating
moviedata.score = (v / (v + m)) * R + (m / (v + m)) * C

#	Recommend 15 movies.
moviedata.sort_values(by='score', ascending=False, inplace=True)
moviedata.head(15)[['title', 'meanRating', 'numVotes', 'score']]

# Separate the year from title.
for i in moviedata.index :
	moviedata.at[i, 'year'] = int(moviedata.at[i, 'title'][-5:-1])
	moviedata.at[i, 'title'] = moviedata.at[i, 'title'][0:-7]

moviedata.to_csv('simple_recommendations.csv', index=False)