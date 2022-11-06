import cv2
import numpy as np
import matplotlib.pyplot as plt


def edge_det(img, filter_type='sobel'):
    img = img.copy()
    img = np.array(img * 255, dtype=np.uint8)
    if filter_type == 'sobel':
        img = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
    elif filter_type == 'laplacian':
        img = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    elif filter_type == 'canny':
        img = cv2.Canny(img, threshold1=100, threshold2=200)
    img = np.array(img / 255, dtype=np.float32)
    return img

def task1(img_path):
    img = cv2.imread(img_path)
    img = img / 255

    img_sobel = edge_det(img, 'sobel')
    img_laplacian = edge_det(img, 'laplacian')
    img_canny = edge_det(img, 'canny')

    cv2.imshow('origin', img)
    cv2.imshow('sobel', img_sobel)
    cv2.imshow('laplacian', img_laplacian)
    cv2.imshow('canny', img_canny)
    cv2.waitKey()


def task2(img_path):
    img = cv2.imread(img_path)
    img = img / 255

    x_grad = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    y_grad = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title('origin')
    plt.subplot(1, 4, 2)
    plt.imshow(x_grad)
    plt.title('x_grad')
    plt.subplot(1, 4, 3)
    plt.imshow(y_grad)
    plt.title('y_grad')
    plt.subplot(1, 4, 4)
    plt.imshow(x_grad + y_grad)
    plt.title('x_grad + y_grad')
    plt.savefig('grad.png')
    plt.show()


def task3(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255

    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
    img_add = img + laplacian
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('origin')
    plt.subplot(1, 3, 2)
    plt.imshow(laplacian)
    plt.title('laplacian')
    plt.subplot(1, 3, 3)
    plt.imshow(img_add)
    plt.title('origin + laplacian')
    plt.savefig('laplacian.png')
    plt.show()



if __name__ == '__main__':
    img_path = '../imgs/bb.png'
    # task1(img_path)
    # task2(img_path)
    # task3(img_path)




