import cv2

def pyr_up_down(img_path):
    img = cv2.imread(img_path)

    cv2.imshow('img', img)
    # cv2.waitKey(0)
    print('img.shape:', img.shape)

    up_img = cv2.pyrUp(img)
    print('up_img.shape:', up_img.shape)
    cv2.imshow('up_img', up_img)

    down_img = cv2.pyrDown(img)
    print('down_img.shape:', down_img.shape)
    cv2.imshow('down_img', down_img)

    cv2.waitKey()


if __name__ == '__main__':
    img_path = '../imgs/bb2.png'
    pyr_up_down(img_path)