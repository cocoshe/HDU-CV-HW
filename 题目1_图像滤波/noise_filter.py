import cv2
import numpy as np


def add_noise(img, noise_type='salt_pepper'):
    img = img.copy()
    if noise_type == 'salt_pepper':
        n = int(0.05 * img.shape[0] * img.shape[1])
        for k in range(n):
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            img[j, i, 0] = 1
            img[j, i, 1] = 1
            img[j, i, 2] = 1
    elif noise_type == 'gaussian':
        size = img.shape
        mean = 0
        sigma = 0.3
        gauss = np.random.normal(mean, sigma, size)
        img = img + gauss
    return img


def add_filter(img, filter_type='median'):
    img = img.copy()
    # print(img)
    if filter_type == 'median':
        img = np.array(img * 255, dtype=np.uint8)
        img = cv2.medianBlur(img, ksize=3)
        img = np.array(img / 255, dtype=np.float32)
    elif filter_type == 'gaussian':
        img = cv2.GaussianBlur(img, (3, 3), 5)
    return img


def noise_filter(img_path):
    img = cv2.imread(img_path)
    img = img / 255

    # 椒盐噪声和中值滤波
    # img_noise = add_noise(img, 'salt_pepper')
    # img_filter = add_filter(img_noise, 'median')

    # 高斯噪声和高斯滤波
    img_noise = add_noise(img, 'gaussian')
    img_filter = add_filter(img_noise, 'gaussian')

    cv2.imshow('origin', img)
    cv2.imshow('noise', img_noise)
    cv2.imshow('filter', img_filter)

    img_origin_np = np.array(img)
    img_filter_np = np.array(img_filter)
    rmse = np.sqrt(np.mean((img_origin_np - img_filter_np) ** 2))
    print('RMSE(after normalization):', rmse)
    print('RMSE(before normalization):', rmse * 255)

    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = '../imgs/bb.png'
    noise_filter(img_path)
