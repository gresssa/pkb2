from mnist import MNIST
import numpy as np 

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
