import numpy as np
import scipy.sparse
from mnist.loader import MNIST

mndata = MNIST('./data')
images, labels = mndata.load_training()
t_images, t_labels = mndata.load_testing()

traiY = labels
trainY = np.array(traiY)

traiX = images
trainX = np.array(traiX)

tesY = t_labels
testY = np.array(tesY)

tesX = t_images
testX = np.array(tesX)
#print(trainY.shape[1], len(np.unique(trainY)))

def getLoss(w,trainX,trainY,lam):
    m = trainX.shape[0]
    trainY_mat = oneHotIt(trainY)
    scores = np.dot(trainX,w)
    prob = softmax(scores)
    loss = (-1 / m) * np.sum(trainY_mat * np.log(prob)) + (lam/2)*np.sum(w*w)
    grad = (-1 / m) * np.dot(trainX.T,(trainY_mat - prob)) + lam*w
    return loss,grad
    
def softmax(z):
    z -= np.max(z)
    smx = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return smx
    
def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds
    
def oneHotIt(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    OHX = np.array(OHX.todense()).T
    return OHX
    
def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy

w = np.zeros([trainX.shape[1],len(np.unique(trainY))])
lam = 1
iterations = 10
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,trainX,trainY,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
    
print ('Training Accuracy: ', getAccuracy(trainX,trainY))
print ('Test Accuracy: ', getAccuracy(testX,testY))