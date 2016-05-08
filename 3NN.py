import idx2numpy as inp
from matplotlib import pyplot as plt
import numpy as np
import math

def sigmoid(x):
    try:
        y = 1 / (1+math.exp(-x))
    except OverflowError:
        y = 0.0
    return y

#-- Reading the images from idx files--
trimgs = inp.convert_from_file('train-images.idx3-ubyte')
trlbls = inp.convert_from_file('train-labels.idx1-ubyte')
tsimgs = inp.convert_from_file('t10k-images.idx3-ubyte')
tslbls = inp.convert_from_file('t10k-labels.idx1-ubyte')
print 'dimensions of train data,(#images,rows,cloumns): ',trimgs.shape
print 'train labels dimensions,(#images labels): ',trlbls.shape
print 'dimensions of test data,(#images,rows,cloumns): ',tsimgs.shape
print 'test labels dimensions,(#images labels): ',tslbls.shape

'''
#-- to Display a image and its lable--
print trlbls[0]
plt.imshow(trimgs[0],interpolation='nearest',cmap=plt.get_cmap('gray'))
plt.show()
'''
# -- CREATION OF OUTPUT VECTORS --
lbvecs = np.zeros((60000,10))
for i in range(60000):
    l = trlbls[i]
    lbvecs[i][l]=1
del trlbls
# -- NORMALIZATION OF INPUT VECTORS --
trimgs = np.resize(trimgs,(60000,784))
trimgs1 = []
for v in trimgs:
    trimgs1.append(np.insert(v,0,1))
trimgs = np.array(trimgs1)
del trimgs1
print "shape of imgages after augmentation",trimgs.shape

tsimgs = np.resize(tsimgs,(10000,784))
tsimgs1 = []
for v in tsimgs:
    tsimgs1.append(np.insert(v,0,1))
tsimgs = np.array(tsimgs1)
del tsimgs1

#-----------------------------------------------------------------------------
eta = 0.005
itrs = 100
'''#-- feed forward ------'''
wji = np.random.uniform(low=-1.0,high=1.0,size=(20,785))
wkj = np.random.uniform(low=-1.0,high=1.0,size=(10,21))
y =np.zeros(21)
y[0]=1
netj = np.zeros(20)
netk = np.zeros(10)
z = np.zeros(10)
sensitivityk = np.zeros(10)
sensitivityj = np.zeros(20)
'''for i in range(20):
    netj[i] = np.dot(wji[i],x)
    y[i+1] = sigmoid(netj[i])
for i in range(10):
    netk[i] = np.dot(wkj[i],y)
    z[i] =sigmoid(netk[i])
'''
''' Back propagation'''

for itr in range(itrs):#iterations
    wjk = wkj.transpose()[1:]
    for i in range(60000):#loop on images
        ''' feed forward'''
        x = trimgs[i]
        t = lbvecs[i]
        for j in  range(20):#input to hidden, hidden units 20
            netj[j] = np.dot(wji[j],x)
            y[j+1] = sigmoid(netj[j])
        for k in range(10):#hidden to output, output units 10
            netk[k] = np.dot(wkj[k],y)
            z[k] = sigmoid(netk[k])
            sensitivityk[k]=(t[k]-z[k])*z[k]*(1-z[k])
            wkj[k]= wkj[k] + eta * y * sensitivityk[k]
        for j in range(20):
            sensitivityj[j]=y[j+1]*(1-y[j+1])*np.dot(wjk[j],sensitivityk)
            wji[j] = wji[j] + eta * x * sensitivityj[j]
    c =np.zeros(10000)
    for i in range(10000):
        x = tsimgs[i]
        for j in range(20):
            netj[j] = np.dot(wji[j],x)
            y[j+1] = sigmoid(netj[j])
        for k in range(10):
            netk[k]=np.dot(wkj[k],y)
            z[k] = sigmoid(netk[k])
        c[i]=np.argmax(z)
    corr =0
    for i in range(10000):
        if c[i]==tslbls[i]:
            corr+=1
    accuracy = corr/100
    print "iteration and accuracy",itr,accuracy
    #if accuracy==84:
        #print "iteration and accuracy:",itr,accuracy
        #break

confmat = np.zeros((10,10))
for i in range(10000):
    confmat[c[i],tslbls[i]] +=1
print "confusion matrix (rows:Predicted,columns:Givenclass)"
print confmat.astype(int)




















