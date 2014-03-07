import os
import struct
from array import array
import numpy as np
from modshogun import MulticlassLabels,RealFeatures
from modshogun import KNN, EuclideanDistance

class Number_recognition:
    def __init__(self, test_images, test_labels, k):
        self.test_images = test_images;
        self.test_labels = test_labels;
        self.k = k;

    def load(self, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                    'got %d' % magic)

            labels = array("B", file.read())

        labels_result = np.zeros(shape=(size))


        for i in xrange(size):
            labels_result[i] = labels[i]

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            print "rows: " + str(rows) + "  cols: " + str(cols)
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                        'got %d' % magic)
            image_data = array("B", file.read())

        images = np.zeros(shape=(size,rows*cols))

        for i in xrange(size):
            images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]


        return images, labels_result

    def load_train(self):
        ims, labels = self.load( self.test_images, self.test_labels)

        self.test_images = ims
        self.test_labels = labels
        labels_numbers = MulticlassLabels(self.test_labels)
        feats  = RealFeatures(self.test_images.T)
        dist = EuclideanDistance()
        self.knn = KNN(self.k, dist, labels_numbers)
        self.knn.train(feats)

    def predict(self, image):
        feats_test  = RealFeatures(image. T)
        pred = self.knn.apply_multiclass(feats_test)
        return pred[:]

