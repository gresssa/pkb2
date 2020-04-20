import numpy as np

def print_statistics(images, t_images, labels, t_labels):
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
        
    print("there are", images.shape[1],"features")
    return
        
def hypothesis(num_class,images,labels,theta):
    h_final=[]
    for i in range(num_class):
        h_0=np.dot(theta[:,i], images[0].T)
        h_0=np.sum(h_0)
        h_final.append(h_0)
    
    h_final=np.array([h_final])
    h_final=h_final.T
    return h_final
    
    
    

