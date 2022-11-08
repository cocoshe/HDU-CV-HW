import cv2
import numpy as np
import matplotlib.pyplot as plt


def edge_det(img, filter_type='sobel', kernel_size=3):
    img_ = img.copy()
    if filter_type == 'sobel':
        img_ = cv2.Sobel(img_, cv2.CV_64F, 1, 1, ksize=kernel_size)
        img_ = cv2.convertScaleAbs(img_)
    elif filter_type == 'laplacian':
        img_ = cv2.Laplacian(img_, cv2.CV_64F, ksize=kernel_size)
        img_ = cv2.convertScaleAbs(img_)
    elif filter_type == 'canny':
        img_ = cv2.Canny(img_, threshold1=100, threshold2=200)
    return img_


def task1(img_path, kernel_size=3):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_origin = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_sobel = edge_det(img, 'sobel', kernel_size)
    img_laplacian = edge_det(img, 'laplacian', kernel_size)
    img_canny = edge_det(img, 'canny')

    plt.figure(figsize=(10, 3))
    plt.subplots_adjust(wspace=0.3, bottom=0.2, top=0.8)
    # plt.suptitle('kernel size: {}'.format(kernel_size), fontsize=26)
    plt.subplot(1, 4, 1)
    plt.imshow(img_origin)
    plt.title('origin')
    plt.subplot(1, 4, 2)
    plt.imshow(img_sobel, cmap='gray')
    plt.title('sobel')
    plt.subplot(1, 4, 3)
    plt.imshow(img_laplacian, cmap='gray')
    plt.title('laplacian')
    plt.subplot(1, 4, 4)
    plt.imshow(img_canny, cmap='gray')
    plt.title('canny')
    plt.savefig('edge_det.png')
    plt.show()



def task2(img_path, kernel_size=3):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_origin = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = img / 255

    x_grad = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    y_grad = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)

    x_grad = cv2.convertScaleAbs(x_grad)
    y_grad = cv2.convertScaleAbs(y_grad)
    img_sobel = cv2.addWeighted(x_grad, 0.5, y_grad, 0.5, 0)



    plt.figure(figsize=(10, 3))
    plt.subplots_adjust(wspace=0.3, bottom=0.2, top=0.8)
    # plt.suptitle('kernel size: {}'.format(kernel_size), fontsize=26)
    plt.subplot(1, 4, 1)
    plt.imshow(img_origin)
    plt.title('origin')
    plt.subplot(1, 4, 2)
    plt.imshow(x_grad, cmap='gray')
    plt.title('x_grad')
    plt.subplot(1, 4, 3)
    plt.imshow(y_grad, cmap='gray')
    plt.title('y_grad')
    plt.subplot(1, 4, 4)
    plt.imshow(img_sobel, cmap='gray')
    plt.title('xy_grad')
    plt.savefig('edge_det.png')
    plt.show()

    # cv2.imshow('x_grad', x_grad)
    # cv2.imshow('y_grad', y_grad)
    # cv2.imshow('x_grad + y_grad', x_grad + y_grad)
    # cv2.waitKey()


def task3(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_origin = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)
    laplacian_ = np.array([laplacian, laplacian, laplacian]).transpose(1, 2, 0)
    # img_add = cv2.addWeighted(img_origin, 1, laplacian_, 1, 0)
    img_add = cv2.add(img_origin, laplacian_)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_origin)
    plt.title('origin')
    plt.subplot(1, 3, 2)
    plt.imshow(laplacian, cmap='gray')
    plt.title('laplacian')
    plt.subplot(1, 3, 3)
    plt.imshow(img_add)
    plt.title('origin + laplacian')
    plt.savefig('laplacian.png')
    plt.show()



if __name__ == '__main__':
    img_path = '../imgs/bb3.png'
    # img_path = '../imgs/gadot.png'
    # task1(img_path, kernel_size=3)
    # task2(img_path, kernel_size=3)
    task3(img_path)




