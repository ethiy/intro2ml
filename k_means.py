# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:54:32 2018

@author: Oussama Ennafii
"""

import skimage
import skimage.io
import skimage.color

import matplotlib.pyplot as plt

import numpy as np
import scipy.spatial.distance

import moviepy.editor

import random
import itertools

import time


def read_image(filename):
    return skimage.io.imread(filename)
    

def sample(image, k):
    l, w = image.shape[:2]
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
    cluster_map = np.full(image.shape[:2], -1, dtype = np.int64)
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
    l, w = image.shape[:2]
    for i, j in itertools.product(range(l), range(w)):
        cluster_map[i, j], cluster_cardinals, clusters = update_pixel(
            i,
            j,
            cluster_map,
            image,
            clusters,
            cluster_cardinals
        )
        if min(cluster_cardinals) <=0:
            print cluster_cardinals
            print clusters
            plt.imshow(cluster_map)
            plt.show()
    return (
        cluster_map,
        clusters,
        cluster_cardinals
    )
    

def barycenter(image):
    return np.mean(image)


def intra_inertia(image, clusters, cluster_cardinals):
    g = barycenter(image)
    return sum(
        [n * distance(gk, g) for n, gk in zip(cluster_cardinals, clusters)]
    )


def k_means(image, k, iterations=1, epsilon=0):
    cluster_map, clusters = initiate(image, k)
    cluster_cardinals = k * [1]
    if epsilon != 0:
        cluster_maps = []
        Ib = [float('inf')]
        iteration = 1
        while True:
            cluster_map, clusters, cluster_cardinals = update_pixels(
                    image,
                    clusters,
                    k * [1],
                    cluster_map
                )
            cluster_maps.append(cluster_map)
            Ib.append(intra_inertia(image, clusters, cluster_cardinals))
            print '-> Iteration :', iteration
            print '---> inertia :', Ib[-1]
            iteration += 1
            if abs(Ib[-2] - Ib[-1]) < epsilon:
                break
    cluster_maps = iterations * [None]
    Ib = iterations * [None]
    for iteration in range(iterations):
        print clusters
        print cluster_cardinals
        
        cluster_map, clusters, cluster_cardinals = update_pixels(
            image,
            clusters,
            cluster_cardinals,
            cluster_map
        )
        cluster_maps[iteration] = cluster_map
        Ib[iteration] = intra_inertia(image, clusters, cluster_cardinals)
        print '-> Iteration :', iteration
        print '---> inertia :', Ib[iteration]
        print '---> population :', cluster_cardinals
    return (cluster_maps, Ib)
    
    
def main():
    image = read_image('complexite.jpg')
    print 'Image size: ', image.shape[:2]
    k = 5
    print 'There are', k, 'clusters.'
    start = time.time()
    cluster_maps, Ib = k_means(image, k, iterations=2, epsilon=0)
    print '-->', time.time()-start, 'seconds'
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(skimage.color.label2rgb(cluster_maps[-1]))
    ax1.set_title('K-means result for k =' + str(k))
    ax2.plot(Ib)
    ax2.set_yscale('log')
    ax2.set_title('Intra-class inertia Ib.')
    moviepy.editor.ImageSequenceClip(
        [skimage.color.label2rgb(cm) for cm in cluster_maps],
        fps=1
    ).write_videofile('k-means_iterations.mp4', fps = 1)
    plt.show()


if __name__ == '__main__':
    main()