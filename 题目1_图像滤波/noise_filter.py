import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

log_rmse = []


def add_noise(img, noise_type='salt_pepper', noise_ratio=0.1, mean=0, sigma=0.3):
    img = img.copy()
    if noise_type == 'salt_pepper':
        n = int(noise_ratio * img.shape[0] * img.shape[1])
        for k in range(n):
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            img[j, i, 0] = 1
            img[j, i, 1] = 1
            img[j, i, 2] = 1
    elif noise_type == 'gaussian':
        size = img.shape
        mean = mean
        sigma = sigma
        gauss = np.random.normal(mean, sigma, size)
        img = img + gauss
    return img


def add_filter(img, filter_type='median', kernel_size=3):
    img = img.copy()
    # print(img)
    if filter_type == 'median':
        img = np.array(img * 255, dtype=np.uint8)
        img = cv2.medianBlur(img, ksize=kernel_size)
        img = np.array(img / 255, dtype=np.float32)
    elif filter_type == 'gaussian':
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 5)
    return img


def noise_filter(img_path, noise_type='salt_pepper', filter_type='median', kernel_size=3, noise_ratio=0.1, log=False, mean=0, sigma=0.3):
    img = cv2.imread(img_path)
    img = img / 255

    # 椒盐噪声和中值滤波
    # img_noise = add_noise(img, 'salt_pepper')
    # img_filter = add_filter(img_noise, 'median')

    # 高斯噪声和高斯滤波
    img_noise = add_noise(img, noise_type, noise_ratio=noise_ratio)
    img_filter = add_filter(img_noise, filter_type, kernel_size)

    # cv2.imshow('origin', img)
    # cv2.imshow('noise', img_noise)
    # cv2.imshow('filter', img_filter)

    img_origin_np = np.array(img)
    img_filter_np = np.array(img_filter)
    rmse = np.sqrt(np.mean((img_origin_np - img_filter_np) ** 2))
    print('RMSE(after normalization):', rmse)
    print('RMSE(before normalization):', rmse * 255)
    if log:
        log_rmse.append(rmse * 255)
    # cv2.waitKey(0)

    img = np.array(img * 255, dtype=np.uint8)
    img_noise = np.array(img_noise * 255, dtype=np.uint8)
    img_filter = np.array(img_filter * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_noise = cv2.cvtColor(img_noise, cv2.COLOR_BGR2RGB)
    img_filter = cv2.cvtColor(img_filter, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(13, 6))
    # plt.figure(figsize=(13, 4))
    plt.subplots_adjust(left=0.05, right=0.95, top=1.1, bottom=-0.2)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('origin.shape: {}'.format(img.shape))
    plt.subplot(1, 3, 2)
    plt.imshow(img_noise)
    plt.title('noise.shape: {}'.format(img_noise.shape))
    plt.subplot(1, 3, 3)
    plt.imshow(img_filter)
    plt.title('filter.shape: {}'.format(img_filter.shape))
    if filter_type == 'median':
        plt.suptitle('kernel_size:({} x {})     noise_ratio:{:.3f}%      RMSE:{:.3f}'.format(kernel_size, kernel_size, noise_ratio * 100, rmse * 255), fontsize=26)
    elif filter_type == 'gaussian':
        plt.suptitle('kernel_size:({} x {})     mean: {}    sigma: {}   RMSE:{:.3f}'.format(kernel_size, kernel_size, mean, sigma,  rmse * 255), fontsize=26)
    plt.savefig('noise_filter.png')
    plt.show()


# def summary(img_path):
#     x, y = np.arange(3, 19, 2), np.arange(0.1, 0.5, 0.05)
#     # x, y = np.arange(1, 5, 2), np.arange(0.1, 0.2, 0.05)
#     for i in x:
#         for j in y:
#             noise_filter(img_path, noise_type='salt_pepper', filter_type='median', kernel_size=i, noise_ratio=j, log=True)
#     ax = plt.subplot(111, projection='3d')
#
#     X, Y = np.meshgrid(x, y)
#     X, Y = X.flatten(), Y.flatten()
#     # X = X[::-1]
#     # Y = Y[::-1]
#     ax.bar3d(X, Y, np.zeros(X.shape), 1, 0.05, log_rmse, shade=True, alpha=0.5)
#     ax.set_xlabel('kernel_size')
#     ax.set_ylabel('noise_ratio')
#     ax.set_zlabel('RMSE')
#     plt.show()



if __name__ == '__main__':
    img_path = '../imgs/bb.png'
    # noise_filter(img_path, noise_type='salt_pepper', filter_type='median', kernel_size=3, noise_ratio=0.3)
    noise_filter(img_path, noise_type='gaussian', filter_type='gaussian', kernel_size=3, mean=0, sigma=1.3)
    # summary(img_path)

