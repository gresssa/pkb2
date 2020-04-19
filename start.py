import numpy as np
import scipy.sparse
from mnist.loader import MNIST

mndata = MNIST('./data')
images, labels = mndata.load_training()
t_images, t_labels = mndata.load_testing()
