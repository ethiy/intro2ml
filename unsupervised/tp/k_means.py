# -*- coding: utf-8 -*-

import os

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


images_dir = '../../resources/images/tp/unsupervised'


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
    cluster_map = np.full(image.shape[:2], -1, dtype=np.int64)
    for c, (i, j) in enumerate(init_indices):
        cluster_map[i, j] = c
    return (cluster_map, clusters)


def assign_pixel(i, j, cluster_map, image, clusters):
    cluster_map[i, j] = np.argmin(
        [distance(image[i, j], center) for center in clusters]
    )
    return cluster_map


def update_pixels(cluster_map, image, clusters):
    l, w = cluster_map.shape
    for i, j in itertools.product(range(l), range(w)):
        cluster_map = assign_pixel(i, j, cluster_map, image, clusters)
    clusters, cluster_cardinals = zip(
        *[
            (
                barycenter(image[cluster_map == center]),
                np.sum(cluster_map == center)
            )
            for center in range(len(clusters))
        ]
    )
    print cluster_cardinals
    print np.sum(cluster_cardinals) == l*w
    return (
        cluster_map,
        clusters,
        intra_inertia(
            image,
            clusters,
            cluster_cardinals
        )
    )


def barycenter(image):
    return np.mean(image)


def intra_inertia(image, clusters, cluster_cardinals):
    g = barycenter(image)
    return sum(
        [nk * distance(gk, g) for nk, gk in zip(cluster_cardinals, clusters)]
    )


def k_means(image, k, iterations=1, epsilon=0):
    cluster_map, clusters = initiate(image, k)
    iteration = 0
    Ibs = [0]
    cluster_maps = []
    while True:
        cluster_map, clusters, ib = update_pixels(
            cluster_map,
            image,
            clusters
        )
        iteration += 1
        print clusters
        print ib
        cluster_maps.append(cluster_map)
        Ibs.append(ib)
        print '----> iteration: ', iteration
        if iteration == iterations or abs(Ibs[-2] - Ibs[-1]) <= epsilon:
            break
    return (cluster_maps, Ibs)


def main():
    image = read_image(os.path.join(images_dir, 'complexite_couleur.jpg'))
    print 'Image size: ', image.shape[:2]
    k = 6
    print 'There are', k, 'clusters.'
    start = time.time()
    cluster_maps, Ibs = k_means(image, k, iterations=50, epsilon=0)
    print '-->', time.time()-start, 'seconds'
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(image)
    ax2.imshow(skimage.color.label2rgb(cluster_maps[-1]))
    ax2.set_title('K-means result for k =' + str(k))
    ax3.plot(Ibs)
    ax3.set_title('Intra-class inertia Ib.')
    moviepy.editor.ImageSequenceClip(
        [skimage.color.label2rgb(cm) for cm in cluster_maps],
        fps=1
    ).write_videofile('k-means_iterations.mp4', fps=1)
    plt.show()


if __name__ == '__main__':
    main()
