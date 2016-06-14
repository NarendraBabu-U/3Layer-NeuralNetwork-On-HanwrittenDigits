# 3Layer-NeuralNetwork-On-HanwrittenDigits

This repository contains code for a complete handwritten digit recognizer on the MNIST digit dataset.

**DATASET :** 
>Downloaded from "http://yann.lecun.com/exdb/mnist"
  
    Contains Handwritten digits in image format.
  
    The digits have been size normalized and centered in fixed-size images.
  
    This is a subset of a larger set available from NIST.
  
    training set: 60,000 samples.
  
    test set: 10,000 samples.

**CODE:**
  
    5-fold cross-validation used for all experiments.
  
    3NN.py : code for 3 layer neural network.

    weightdecay.py : code for 3 layer weight decay version.
  
    Manufacturingdata_noise.py : code for 3 layer manufacturing data  (Gaussian noise added) version.
  
    knn.py : code for nearest neighbour (k=1).
