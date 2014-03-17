#!/usr/bin/env python

# This example is about learning and using eigenfaces in Shogun. We demonstrate how to use them for a set of faces.

import cv2
import numpy as np

from modshogun import RealFeatures
from modshogun import PCA
from modshogun import EuclideanDistance
from numpy import linalg as LA

import math
import os

IMAGE_WIDHT = 25
IMAGE_HEIGHT = 25

class EigenFaces():
    def __init__(self, num_components):
        """
        Constructor
        """
        self._num_components = num_components;
        self._projections = []

    def train(self, images, labels):
        """
        Train eigenfaces
        """
        print "Train..."
        #copy labels
        self._labels = labels;

        #transform the numpe vector to shogun structure
        features = RealFeatures(images)
        #PCA
        self.pca = PCA()
        # set dimension
        self.pca.set_target_dim(self._num_components);
        # compute PCA     
        self.pca.init(features)
       
        for sampleIdx in range(features.get_num_vectors()):
            v = features.get_feature_vector(sampleIdx);
            p = self.pca.apply_to_feature_vector(v);
            self._projections.insert(sampleIdx, p);

        print "Train ok!"

    def predict(self, image):
        """
        Predict the face
        """
        # image as row
        imageAsRow = np.asarray(image.reshape(image.shape[0]*image.shape[1],1), np.float64);
        # project inthe subspace
        p = self.pca.apply_to_feature_vector(RealFeatures(imageAsRow).get_feature_vector(0));

        minDist =1e100; # min value to find the face
        minClass = -1;  #class      
        #search which face is the best match
        for sampleIdx in range(len(self._projections)):
            test = RealFeatures(np.asmatrix(p,np.float64).T)
            projection = RealFeatures(np.asmatrix(self._projections[sampleIdx],np.float64).T)
            dist = EuclideanDistance( test, projection).distance(0,0)

            if(dist < minDist ):
                minDist = dist;
                minClass = self._labels[sampleIdx];

        return minClass

    def getMean(self):
        """
        Return the mean vector
        """
        return self.pca.get_mean()

    def getEigenValues(self):
        """
        Return the eigenvalues vector
        """
        return self.pca.get_eigenvalues();

def readImages(list_filenames):
    """
    Read all the image. Image as rows
    """
    print "Reading images ..."
    # reserve space for the matrix
    images = np.empty( (IMAGE_HEIGHT*IMAGE_WIDHT, (len(list_filenames))-1))
    index = 0;
    for im_filename in list_filenames:
        # read image with opencv
        imagen= cv2.imread(im_filename, cv2.IMREAD_GRAYSCALE)
        # resize image -> problem with PCA N>>D
        imagen = cv2.resize(imagen, (IMAGE_HEIGHT, IMAGE_WIDHT));
        images[:,index] = imagen.reshape(imagen.shape[0]*imagen.shape[1],1).T;
        index=index + 1
        # don't read the last value (last value is to test eigenfaces)         
        if( (len(list_filenames)-1)==index):
            break
    print "OK! " 
    return images

# contains images (path) and labels
# DATABASE: AT&T Facedatabase
def get_imlist(path, NUM_PERSONS, NUM_IMAGES_PER_PERSON):

    """ Returns a list of filenames for NUM_PERSONS and NUM_IMAGES_PER_PERSON """
    list_filenames=[]
    list_labels=[]
    num_person = 0
    #list all the directories
    for directory in os.listdir(path): 
        relative_path =path+'/'+directory

        if NUM_PERSONS==num_person:
            break;
        #is directory?
        if os.path.isdir(relative_path):
            num_image = 0;
            for f in os.listdir(relative_path):
                os.path.join(relative_path,f)
                # is an image?                
                if f.endswith('.pgm'):
                    num_image = num_image +1
                    list_labels.append(num_person)
                    list_filenames.append(relative_path+'/'+f)  
                    if num_image==NUM_IMAGES_PER_PERSON:
                        break
        num_person = num_person+1
    return [list_filenames, list_labels]

if __name__ == '__main__':
    # return list of filenames and labels
    [list_filenames, list_labels] = get_imlist('../../att_faces', 40, 10)

    #read all images
    images = readImages(list_filenames);

    #  this class resolves the eigenfaces
    eigenfaces = EigenFaces(100)
    #############################################
    # train eigenfaces
    #############################################
    eigenfaces.train(images, list_labels)

    #############################################
    # test eigenfaces
    #############################################
    image = cv2.resize(cv2.imread(list_filenames[-1], cv2.IMREAD_GRAYSCALE), (IMAGE_HEIGHT, IMAGE_WIDHT));
    print "predicted: ", eigenfaces.predict(image), " // real: " ,list_labels[-1] 


    #############################################
    #   Mean face
    #############################################
    # get mean and reshape ( height and width original size)
    mean = eigenfaces.getMean().reshape(IMAGE_HEIGHT, IMAGE_WIDHT);
    # create mean normalize image (0, 255, type = uint8)   
    mean_normalize = np.zeros((IMAGE_HEIGHT, IMAGE_WIDHT,1), np.uint8)
    # normalize
    mean_normalize= cv2.normalize(mean, mean_normalize, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1);
    #show
    cv2.imshow("mean_normalize", mean_normalize)
    cv2.waitKey(0)


    #############################################
    #   Reconstruction with diferents values of eigenvectos
    #############################################
    # Read the last image of the file to test Eigenfaces
    image = cv2.resize(cv2.imread(list_filenames[0], cv2.IMREAD_GRAYSCALE), (IMAGE_HEIGHT, IMAGE_WIDHT));
    # image as row
    imageAsRow = np.asarray(image.reshape(image.shape[0]*image.shape[1],1), np.float64);

    # Reconstruct 10 eigen vectors to 300, step 15
    for i in range(10, 300, 50):

        print "Reconstruct with " + str(i) + " eigenvectors" 

        pca = PCA()
        # set dimension
        pca.set_target_dim(i);
        # compute PCA     
        pca.init(RealFeatures(images))

        pca.apply_to_feature_vector(RealFeatures(imageAsRow).get_feature_vector(0));

        #reconstruct
        projection = pca.apply_to_feature_vector(RealFeatures(imageAsRow).get_feature_vector(0));

        reconstruction = np.asmatrix( np.asarray(projection, np.float64))*np.asmatrix(pca.get_transformation_matrix()).T
        reconstruction = reconstruction + pca.get_mean()

        #normlize image
        reconstruction_normalize = np.zeros((IMAGE_HEIGHT, IMAGE_WIDHT,1), np.uint8)
        reconstruction_normalize = cv2.normalize(reconstruction, reconstruction_normalize, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1);
        reconstruction_normalize = reconstruction_normalize.reshape(IMAGE_HEIGHT, IMAGE_WIDHT)
        # show reconstruction        
        cv2.imshow("reconstruction" + str(i),reconstruction_normalize)
    cv2.waitKey(0)


