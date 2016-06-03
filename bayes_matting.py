from skimage import data, io, filters, novice
from scipy import ndimage
from numpy.linalg import inv, LinAlgError

import numpy as np


def bayes_matting(img, trimap, sigma_d=10, it=1):
    """
    Implements Bayes Matting given an image and a trimap.

    :param img:
    :param trimap:
    :param sigma_d:
    :param it:
    :return:
    """
    assert img.shape[:-1] == trimap.shape

    img = img / 255.0
    nrows, ncols = img.shape[:-1]

    # initial alpha guess
    alpha = np.zeros(trimap.shape)
    alpha[trimap == 255] = 1
    alpha[trimap == 128] = 0.5

    B = img[alpha == 0] # background pixel mask
    F = img[alpha == 1] # foreground pixel mask

    mean_B = np.mean(B, axis=0)
    cov_B = np.cov(B.T)
    mean_F = np.mean(F, axis=0)
    cov_F = np.cov(F.T)

    try:
        inv_cov_B = np.linalg.inv(cov_B)
        inv_cov_F = np.linalg.inv(cov_F)
    except LinAlgError:
        print("LinAlgError")

    for i in range(it):
        print("Iteration {}".format(i))
        for row in range(nrows):
            for col in range(ncols):
                if trimap[row, col] == 128:
                    f, g = calculate_fg(img[row, col], alpha[row, col], mean_F, inv_cov_F,  mean_B, inv_cov_B, sigma_d)
                    alpha[row, col] = calculate_alpha(img[row, col], f, g)

        alpha = np.clip(alpha, 0, 1)

    return alpha


def calculate_fg(color, alpha, mean_f , inv_cov_f,  mean_b, inv_cov_b, sigma_d):
    """
    One step in calculating foreground and background

    :param color:
    :param alpha:
    :param mean_f:
    :param cov_f:
    :param mean_b:
    :param cov_b:
    :param sigma_d:
    :return: new Foreground and Background Color
    """

    a_11 = inv_cov_f + (alpha**2 / sigma_d**2) * np.identity(3)
    a_22 = inv_cov_b + ((1 - alpha)**2 / sigma_d**2) * np.identity(3)
    a_12 = a_21 = (alpha * (1 - alpha) / sigma_d**2) * np.identity(3)

    b_1 = np.dot(inv_cov_f, mean_f) + (alpha / sigma_d**2) * color
    b_2 = np.dot(inv_cov_b, mean_b) + ((1 - alpha) / sigma_d**2) * color

    l = np.empty([6,6])
    l[0] = np.append(a_11[0], a_12[0])
    l[1] = np.append(a_11[1], a_12[1])
    l[2] = np.append(a_11[2], a_12[2])
    l[3] = np.append(a_21[0], a_22[0])
    l[4] = np.append(a_21[1], a_22[1])
    l[5] = np.append(a_21[2], a_22[2])
    r = np.append(b_1, b_2)

    return np.split(np.linalg.solve(l, r), 2)


def calculate_alpha(color, F, B):
    """
    One step in calculating new alpha values

    :param color: color value at pixel
    :param F: foreground value at pixel
    :param B: background value at pixel
    :return: alpha value
    """
    return np.dot(color - B, F - B) / np.dot(F - B, F - B)
