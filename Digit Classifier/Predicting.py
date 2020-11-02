# Using the model
# For future work look at the MNIST handwritten digit dataset.
# Weblink- http://yann.lecun.com/exdb/mnist/

import sys
import numpy as np
import pandas as pd
from matplotlib.image import imread

import pickle as pkl
inFile = open('DigitClassifier.pkl', 'rb')
classifier = pkl.load(inFile)
inFile.close()

if len(sys.argv) < 2 :
	print('Error: Please specify an image file to be classified.')
	print(sys.argv)
	sys.exit()
	
fileNames = sys.argv[1:]

images = [imread(file) for file in fileNames]
images[0].shape
#	32 x 32 x 3

grayImages = [img[..., 0] for img in images]
#	32 x 32

binaryImages = [np.around(gray) for gray in grayImages]
reducedImages = [np.array(
	[[bin[row : row + 4, col : col + 4].sum()
	for col in range(0, 32, 4)]
	for row in range(0, 32, 4)])
	for bin in binaryImages]
#	8 x 8 image

# print(classifier.predict(np.expand_dims(reducedImage.ravel(), 0)))
# Can request several predictions with one call.
# Expanded image in another array with horizontal element stacking.
arrayToPredict = [red.ravel() for red in reducedImages]
predictions = classifier.predict(arrayToPredict)

for (file, pred) in zip(fileNames, predictions) :
	print(file + ' ----> ' + str(pred) + '\n')
