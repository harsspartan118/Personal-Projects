# Observing dataset

import matplotlib.pyplot as plt
import sys
from sklearn.datasets import load_digits

dataset = load_digits()
seed = 0
numRows = 5

if len(sys.argv) == 2 :
	seed = int(sys.argv[1])
	seed = seed % 100
if len(sys.argv) == 3 :
	numRows = int(sys.argv[2])
	numRows = numRows if numRows <= 15 else 15

plt.figure(figsize = (10, numRows))
for col in range(10) :
	filteredImages = dataset.images[dataset.target == col]
	for row in range(0, 10 * numRows, 10) :
		plt.subplot(numRows, 10, row + col + 1)
		frame = plt.imshow(filteredImages[seed + row], cmap = 'gray')
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)

plt.show()