import numpy as np
import math
import idx2numpy as inp
from sklearn.neighbors import KNeighborsClassifier

trimgs = inp.convert_from_file('train-images.idx3-ubyte')
trlbls = inp.convert_from_file('train-labels.idx1-ubyte')
tsimgs = inp.convert_from_file('t10k-images.idx3-ubyte')
tslbls = inp.convert_from_file('t10k-labels.idx1-ubyte')
trimgs = np.resize(trimgs,(60000,784))
tsimgs = np.resize(tsimgs,(10000,784))

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(trimgs, trlbls)

pred=neigh.predict(tsimgs)
corr=0
for i in range(10000):
    if (pred[i]==tslbls[i]):
        corr+=1
print "accuracy: ",corr/100

