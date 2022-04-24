from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def init_centroids(num_clusters, image):
    """
    Инициализируйте np-массив размерности `num_clusters` x image_shape[-1] RGB цветами
    случайно выбранных пикселей картинки `image`

    Аргументы
    ----------
    num_clusters : int
        Количество центроидов/кластеров
    image : nparray
        (H, W, C) картинка, представленная в виде np-массива

    Возвращаемое значение
    -------
    centroids_init : nparray
        Случайным образом инициализированные центроиды
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    centroids_init = image[
        np.random.choice(image.shape[0], num_clusters, replace=False),
        np.random.choice(image.shape[1], num_clusters, replace=False)
    ]
    # *** КОНЕЦ ВАШЕГО КОДА ***

    return centroids_init


def get_dist(image_4d: np.array, centroids: np.array):
    norms = np.sqrt(np.sum(np.square(image_4d - centroids), axis=3))
    ret = np.argmin(norms, axis=2)
    return ret


def get_image_4d(image, dim_size):
    new_img = np.array(image)
    new_img = new_img.reshape((image.shape[0], image.shape[1], 1, image.shape[2]))
    new_img = new_img.repeat(dim_size, 2)
    return new_img


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Выполните шаг обновления позиций центроидов алгоритма k-средних `max_iter` раз

    Аргументы
    ----------
    centroids : nparray
        np массив с центроидами
    image : nparray
        (H, W, C) картинка, представленная в виде np-массива
    max_iter : int
        Количество итераций алгоритма
    print_every : int
        Частота вывода диагностического сообщения

    Возвращаемое значение
    -------
    new_centroids : nparray
        Новые значения центроидов
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    start = time.time()
    I = np.eye(centroids.shape[0]).astype(np.bool).reshape((centroids.shape[0], centroids.shape[0], 1))
    image_4d = get_image_4d(image, centroids.shape[0])
    for i in range(max_iter):
        if i % print_every == 0:
            print('Iteration: ' + str(i))
            print(centroids)
        dists = get_dist(image_4d, centroids)
        whitelist_acc = np.r_[I][dists]
        acc = image_4d * whitelist_acc
        nc = np.mean(acc, axis=(0, 1), where=whitelist_acc.repeat(image.shape[2], 3))
        if np.sum(np.abs(nc - centroids)) < 0.00001:
            print('Break at iteration ', i)
            break
        centroids = nc
    end = time.time()
    print("The time of execution of above program is :", end - start)
    # *** КОНЕЦ ВАШЕГО КОДА ***
    return centroids


def update_image(image, centroids):
    """
    Обновите RGB значения каждого пикселя картинки `image`, заменив его
    на значение ближайшего к нему центроида из `centroids`

    Аргументы
    ----------
    image : nparray
        (H, W, C) картинка, представленная в виде np-массива
    centroids : nparray
        np массив с центроидами

    Возвращаемое значение
    -------
    image : nparray
        Обновленное изображение
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    # def compute_min_dist(pixel):
    #     nonlocal centroids
    #     norm_innter = np.square(centroids - pixel)
    #     norm = np.sqrt(np.sum(norm_innter, axis=1))
    #     val = np.argmin(norm)
    #     return centroids[val] / 255
    # return np.apply_along_axis(compute_min_dist, 2, image)

    dist = get_dist(get_image_4d(image, centroids.shape[0]), centroids)
    return np.r_[np.round(centroids).astype(np.uint8)][dist]
    # *** КОНЕЦ ВАШЕГО КОДА ***


def main(args):
    # Setup
    np.random.seed(100)
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
