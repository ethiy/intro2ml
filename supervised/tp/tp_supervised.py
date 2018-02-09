# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import os

images_dir='../../resources/images/tp/supervised'

import numpy as np

import gdal
import gdalconst


import sklearn.neighbors
import sklearn.metrics
import sklearn.mixture
import sklearn.tree


def read(filename):
    """
        reads all bands of raster images.
        
        :param filename: the path to the raster image
        :type filename: string
        :return: a list containing a numpy matrix for each band
        :rtype: list
    """
    dataset = gdal.Open(filename, gdalconst.GA_ReadOnly)
    return np.dstack(
        [
            dataset.GetRasterBand(band).ReadAsArray().astype(np.float64)
            for band in range(1, dataset.RasterCount + 1)
        ]
    )


def separate_data(image, ground_truth, fraction=.7):
    data = read(image)
    gt = read(ground_truth)
    return (
        (
            data[:int(data.shape[0] * fraction),:,:],
            data[int(data.shape[0] * fraction):,:,:]
        ),
        (
            gt[:int(data.shape[0] * fraction),:,:],
            gt[int(data.shape[0] * fraction):,:,:]
        )
    )


def format_data(image, ground_truth, fraction=.7):
    (x_train, x_test), (y_train, y_test) = separate_data(
        image,
        ground_truth,
        fraction
    )
    
    return (
        (
            np.reshape(x_train, (-1, x_train.shape[2])),
            np.reshape(x_test, (-1, x_test.shape[2]))
        ),
        (
            np.reshape(y_train, (-1, y_train.shape[2])),
            np.reshape(y_test, (-1, y_test.shape[2]))
        )
    )

if __name__ == '__main__':
    (x_train, x_test), (y_train, y_test) = format_data(
        os.path.join(images_dir, 'sentinel-2_sample.tif'),
        os.path.join(images_dir, 'ground_truth_forest.tif')
    )
    
    knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5).fit(
        x_train,
        y_train.ravel()
    )
    
    knn_cm = sklearn.metrics.confusion_matrix(
        y_test,
        knn_model.predict(x_test)
    )
    
    print('K-NN confusion matrix with k = 5', knn_cm)
    
    centroid_model = sklearn.neighbors.NearestCentroid().fit(
        x_train,
        y_train.ravel()
    )
    
    centroid_cm = sklearn.metrics.confusion_matrix(
        y_test,
        knn_model.predict(x_test)
    )
    
    print('Centroid classifier confusion matrix', centroid_cm)
    
    gmm_model = sklearn.mixture.GMM(n_components=2).fit(
        x_train,
        y_train.ravel()
    )
    
    gmm_cm = sklearn.metrics.confusion_matrix(
        y_test,
        knn_model.predict(x_test)
    )
    
    print('GMM classifier confusion matrix', gmm_cm)

    dt_model = sklearn.tree.DecisionTreeClassifier(max_depth=10).fit(
        x_train,
        y_train.ravel()
    )
    
    dt_cm = sklearn.metrics.confusion_matrix(
        y_test,
        knn_model.predict(x_test)
    )
    
    print('Decision Tree classifier confusion matrix with max depth = 100', dt_cm)