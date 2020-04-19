import numpy as np
import scipy.sparse
from mnist.loader import MNIST

mndata = MNIST('./data')
images, labels = mndata.load_training()
t_images, t_labels = mndata.load_testing()

images = np.array(images)
t_images = np.array(t_images)
labels = np.array(labels)
t_labels = np.array(t_labels)

print("training instances:", images.shape[0])
print("testing instances:", t_images.shape[0])

num_class = len(np.unique(labels))

print("There are", num_class,"classes")

print("statistics for training")

for i in range(num_class):
    print("class", i, "instances: ", np.sum(labels==i))

print("statistics for testing") 

for i in range(num_class):
    print("class", i, "instances: ", np.sum(t_labels==i))

def getLoss(w,images,labels,lam):
    m = images.shape[0]
    labels_mat = oneHotIt(labels)
    scores = np.dot(images,w)
    prob = softmax(scores)
    loss = (-1 / m) * np.sum(labels_mat * np.log(prob)) + (lam/2)*np.sum(w*w)
    grad = (-1 / m) * np.dot(images.T,(labels_mat - prob)) + lam*w
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

w = np.zeros([images.shape[1],num_class])
lam = 1
iterations = 10
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,images,labels,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
    
print ('Training Accuracy: ', getAccuracy(images,labels))
print ('Test Accuracy: ', getAccuracy(t_images,t_labels))