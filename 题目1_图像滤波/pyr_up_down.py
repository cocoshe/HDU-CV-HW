import cv2
import matplotlib.pyplot as plt


def pyr_up_down(img_path):
    img = cv2.imread(img_path)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    print('img.shape:', img.shape)
    cv2.imwrite('img.png', img)

    up_img = cv2.pyrUp(img)
    print('up_img.shape:', up_img.shape)
    # cv2.imshow('up_img', up_img)
    cv2.imwrite('up_img.png', up_img)

    down_img = cv2.pyrDown(img)
    print('down_img.shape:', down_img.shape)
    # cv2.imshow('down_img', down_img)
    cv2.imwrite('down_img.png', down_img)

    # cv2.waitKey()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    up_img = cv2.cvtColor(up_img, cv2.COLOR_BGR2RGB)
    down_img = cv2.cvtColor(down_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(13, 3))
    plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('origin.shape: {}'.format(img.shape))
    plt.subplot(1, 3, 2)
    plt.imshow(up_img)
    plt.title('up_img.shape: {}'.format(up_img.shape))
    plt.subplot(1, 3, 3)
    plt.imshow(down_img)
    plt.title('down_img.shape: {}'.format(down_img.shape))
    plt.savefig('pyr_up_down.png')
    plt.show()


if __name__ == '__main__':
    img_path = '../imgs/jason.png'
    pyr_up_down(img_path)