# Taken directly from Wayne's repo

import numpy as np
import os

cifar_dir = "data/cifar-10-batches-py"
target_dir = "data/greyscale_cifar"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

training_images = np.zeros([50000,1024])

for i in range(5):
    print(i)
    img = unpickle(os.path.join(cifar_dir,f'data_batch_{i+1}'))
    img = img[b'data']
    R, G, B = img[:,:1024], img[:,1024:2048], img[:,2048:3072]
    training_images[i*10000:(i+1)*10000] = (0.2989 * R + 0.5870 * G + 0.1140 * B)
# training_images = (training_images - training_images.mean(axis=1)[:,None]) / training_images.std(axis=1)[:,None]
training_images = training_images / 255.
np.save(os.path.join(target_dir, 'training_images.npy'),training_images)

img = unpickle(os.path.join(cifar_dir,'test_batch'))
img = img[b'data']
R, G, B = img[:,:1024], img[:,1024:2048], img[:,2048:3072]
test_images = (0.2989 * R + 0.5870 * G + 0.1140 * B)
# test_images = (test_images - test_images.mean(axis=1)[:,None]) / test_images.std(axis=1)[:,None]
test_images = test_images / 255.
np.save(os.path.join(target_dir, 'test_images.npy'),test_images)
