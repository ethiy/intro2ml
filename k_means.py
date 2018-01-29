# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:54:32 2018

@author: Oussama Ennafii
"""

import skimage.io
import skimage.color

import numpy as np
import scipy.spatial.distance

import moviepy.editor

import random
import itertools

import time


def read_image(filename):
    return skimage.io.imread(filename)
    

def sample(image, k):
    l, w = image.shape
    indices = random.sample(
        list(
            itertools.product(
                range(l),
                range(w)
            )
        ),
        k
    )
    return (indices, [image[i, j] for i, j in indices])
    

def distance(lhs, rhs):
    return scipy.spatial.distance.euclidean(lhs, rhs)
    

def initiate(image, k):
    init_indices, clusters = sample(image, k)
    cluster_map = np.full_like(image, -1, dtype = np.int64)
    for c, (i, j) in enumerate(init_indices):
        cluster_map[i, j] = c
    return (cluster_map, clusters)


def update_pixel(i, j, cluster_map, image, clusters, cluster_cardinals):
    ds = [distance(image[i, j], k) for k in clusters]
    new = np.argmin(ds)
    cluster_cardinals[new] += 1
    clusters[new] = (clusters[new] * (cluster_cardinals[new] - 1) + image[i, j]) / cluster_cardinals[new]
    if cluster_map[i, j] > -1:
        clusters[cluster_map[i, j]] = (
            (clusters[cluster_map[i, j]] * cluster_cardinals[cluster_map[i, j]]
            -
            image[i, j]) / (cluster_cardinals[cluster_map[i, j]] - 1)
            if cluster_cardinals[cluster_map[i, j]] > 1 else 0
        )
        cluster_cardinals[cluster_map[i, j]] -= 1
    return (
        new,
        cluster_cardinals,
        clusters
    )


def update_pixels(image, clusters, cluster_cardinals, cluster_map):
    l, w = image.shape
    for i, j in itertools.product(range(l), range(w)):
        cluster_map[i, j], cluster_cardinals, clusters = update_pixel(
            i,
            j,
            cluster_map,
            image,
            clusters,
            cluster_cardinals
        )
    return (
        cluster_map,
        clusters,
        cluster_cardinals
    )


def k_means(image, k, iterations=1, epsilon=0):
    cluster_map, clusters = initiate(image, k)
    if epsilon != 0:
        iterations = 10
    cluster_maps = iterations * [None]
    for iteration in range(iterations):
        cluster_map, clusters, cluster_cardinals = update_pixels(
            image,
            clusters,
            k * [1],
            cluster_map
        )
        cluster_maps[iteration] = cluster_map
        print '--> Iteration :', iteration
    return cluster_maps
    
    
def main():
    image = read_image('label_3_it_17.jpg')
    print 'Il y a:', image.size, 'pixels'
    k = 5
    print 'Le nombre de clusters k =', k
    start = time.time()
    cluster_maps = k_means(image, k, iterations=10, epsilon=0)
    print '-->', time.time()-start, 'seconds'
    moviepy.editor.ImageSequenceClip(
        [skimage.color.label2rgb(cm) for cm in cluster_maps],
        fps=1
    ).write_videofile('test.mp4', fps = 1)


if __name__ == '__main__':
    main()