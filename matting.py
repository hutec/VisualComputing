from skimage import data, io, filters, novice
from scipy import ndimage
from numpy.linalg import inv, LinAlgError

import numpy as np


def blue_screen_matting(a1, a2, img):
    """
    Implements Blue Screen Matting.
    alpha = 1 - alpha_1 (I_b - a_2*I_g)

    param a1: alpha_1
    param a2: alpha_2
    param img: image for matting
    return alpha: alpha map

    """
    assert 0.5 < a2 < 1.5
    
    nrows, ncols, _ = img.shape
    alpha = np.ndarray(shape=(nrows, ncols))
    for row in range(nrows):
        for col in range(ncols):
            Ib = img[row, col, 2] / 255.0
            Ig = img[row, col, 1] / 255.0
            alpha[row, col] = 1 - a1 * (Ib - a2 * Ig)
            
    return np.clip(alpha, 0, 1)


def triangulation_matting(img1, bg1, img2, bg2):
    """
    Implements triangulation where two images with
    the same foreground but differing background are given.
        
    param img1:
    param bg1:
    param img2:
    param bg2:
    """
    assert img1.shape == img2.shape == bg1.shape == bg2.shape
    
    nrows, ncols, _ = img1.shape
    alpha = np.zeros(shape=(nrows, ncols))
    for row in range(nrows):
        for col in range(ncols):
            i1 = img1[row, col] / 255.0
            i2 = img2[row, col] / 255.0
            b1 = bg1[row, col] / 255.0
            b2 = bg2[row, col] / 255.0
            if np.array_equal(b1, b2) or np.dot(b1 - b2, b1 - b2) == 0:
                alpha[row, col] = 0
            else:
                alpha[row, col] = 1 - np.dot(i1 - i2, b1 - b2) / np.dot(b1 - b2, b1 - b2)

    return np.clip(alpha, 0, 1)


def bayes_matting(img, trimap, sigma_d=0.5, it=1, log=False):
    """
    Implements Bayesian Matting.
    Given a background for calculating the mean and variance
    and a foreground image

    param img: composite image 
    param trimap: user segmented trimap
    param sigma_d: parameter that reflects the expected deviation from matting assumption
    param it: number of iterations
    """
    assert img.shape[:-1] == trimap.shape
    
    img = img / 255.0
    nrows, ncols = img.shape[:-1]
    
    # initial alpha guess
    alpha = np.zeros(trimap.shape)
    alpha[trimap == 255] = 1
    alpha[trimap == 128] = 0.5

    alpha_log = []

    B = img[alpha == 0] # background pixel mask
    F = img[alpha == 1] # foreground pixel mask
    
    while it > 0:

        mean_B = np.mean(B, axis=0)
        cov_B = np.cov(B.T)
        mean_F = np.mean(F, axis=0)
        cov_F = np.cov(F.T)

        print("Background pixels: {}, Foreground Pixels: {}".format(len(B), len(F)))
        # print("alpha[430, 500]: {}".format(alpha[430, 500]))
        # print("mean_B: {}, mean_F: {}".format(mean_B, mean_F))

        try:
            inv_cov_B = np.linalg.inv(cov_B)
            inv_cov_F = np.linalg.inv(cov_F)
        except LinAlgError:
            print("LinAlgError")

        F = np.zeros(img.shape)
        B = np.zeros(img.shape)

        for row in range(nrows):
            for col in range(ncols):
                if alpha[row, col] == 1:
                    F[row, col] = img[row, col]
                    B[row, col] = 0
                    continue
                if alpha[row, col] == 0:
                    F[row, col] = 0
                    B[row, col] = img[row, col]
                    continue

                I = img[row, col]
                a = alpha[row, col]

                a_11 = inv_cov_F + (a**2 / sigma_d**2) * np.identity(3)
                a_22 = inv_cov_B + ((1 - a)**2 / sigma_d**2) * np.identity(3)
                a_12 = a_21 = (a * (1 - a) / sigma_d**2) * np.identity(3)

                b_1 = np.dot(inv_cov_F, mean_F) + (a / sigma_d**2) * I
                b_2 = np.dot(inv_cov_B, mean_B) + ((1 - a) / sigma_d**2) * I

                l = np.empty([6,6])
                l[0] = np.append(a_11[0], a_12[0])
                l[1] = np.append(a_11[1], a_12[1])
                l[2] = np.append(a_11[2], a_12[2])
                l[3] = np.append(a_21[0], a_22[0])
                l[4] = np.append(a_21[1], a_22[1])
                l[5] = np.append(a_21[2], a_22[2])
                r = np.append(b_1, b_2)

                F[row, col], B[row, col] = np.split(np.linalg.solve(l, r), 2)

        diff = 0

        for row in range(nrows):
            for col in range(ncols):
                if alpha[row, col] == 1 or alpha[row, col] == 0:
                    continue

                alpha_new = (np.dot((img[row, col] - B[row, col]), (F[row, col] - B[row, col])) /
                                   np.dot(F[row, col] - B[row, col], F[row, col] - B[row, col]))
                diff += np.abs(alpha[row, col] - alpha_new)
                alpha[row, col] = alpha_new

        alpha = np.clip(alpha, 0, 1)

        print("Alpha Diff: {}".format(diff))

        F = F[alpha != 0]
        B = B[alpha != 1]

        if log:
            alpha_log.append(alpha)
        
        it -= 1
    
    #if log:
    #    return F.reshape(img.shape), B.reshape(img.shape), alpha, alpha_log
    #else:
    #    return F.reshape(img.shape), B.reshape(img.shape), alpha

    return alpha
