# Model Building

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix

dataset = load_digits()

# Dictionary content-

list(dataset.keys())
#	['data', 'target', 'target_names', 'images', 'DESCR']

images = dataset['images']
images.shape
# 1797 image matrices (8 x 8) in an ndarray

twos = dataset.images[dataset.target == 2]
plt.imshow(twos[0], cmap = 'gray')
plt.show()
# See images.

max(images.flat)
# Integers ranging from 0 to 16

data = dataset['data']
data.shape
# 1797 image vectors (64) in an ndarray

target = dataset['target']
# 0 - 9 in ndarray. Same as target_names here.
	
open('Dataset desc.txt', 'w').write(dataset['DESCR'])
# Gray scale images
# 32 x 32 images divided into 4 x 4 blocks (total 8 x 8)
# Number of on pixels counted (0 - 16)


xTrain, xTest, yTrain, yTest = train_test_split(data, target, train_size = 0.7, test_size = 0.3)
xCV, xTest, yCV, yTest = train_test_split(xTest, yTest, train_size = 0.67, test_size = 0.33)

lr1 = LogisticRegression(C = 1)
lr1.fit(xTrain, yTrain)
lr1.score(xTrain, yTrain)	# 99% accuracy
lr1.score(xCV, yCV)	# 95% accuracy

lr2 = LogisticRegression(C = 100)
lr2.fit(xTrain, yTrain)
lr2.score(xTrain, yTrain)	# 99.9%
lr2.score(xCV, yCV)	#94%

lr3 = LogisticRegression(C = 0.1)
lr3.fit(xTrain, yTrain)
lr3.score(xTrain, yTrain)	# 98%
lr3.score(xCV, yCV)	# 96%

lr3.score(xTest, yTest)	# 96%

yPredicted = lr3.predict(xTest)
confusion_matrix(yTest, yPredicted)
#	7 errors


[i for i in range(len(yTest)) if yTest[i] != yPredicted[i]]
#	Indexes of error images

xTest[8].shape
#	64

plt.imshow(np.reshape(xTest[8], (8, 8)), cmap = 'gray')
plt.show()
#	See the error image as an image matrix.

import pickle as pkl
outFile = open('DigitClassifier.pkl', 'wb')
pkl.dump(lr3, outFile)
outFile.close()
