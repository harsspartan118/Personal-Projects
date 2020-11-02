# Predicting sentiments

import pickle as pkl

file=open('FinalModel.pkl', 'rb')

# Will load 1.4 GB in memory.
model=pkl.load(file)

tweet1 = "damn.. My life sucks"
tweet2 = "Saw the launch of Sputnik1.. So cool..!"
tweet3 = "started my project today."
tweets = [tweet1, tweet2, tweet3]

def predictSentiment(tweets) :
	if type(tweets) == str :
		tweets = [tweets, ]
	prediction = model[1].predict(model[0].transform(tweets))
	return ['Positive' if prediction[i] != 0 else 'Negative' for i in range(len(tweets))]
	
predictSentiment(tweets)