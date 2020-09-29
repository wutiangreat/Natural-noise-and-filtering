import cv2
from PIL import Image
import numpy as np
global img

def main():
    global img
#    img = Image.open('./test/target_0_base_5_4365.png')
    img = cv2.imread('.//test//003.png',0)  # 手写数字图像所在位置
    retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
#    print(np.array(img))
#    print('hhh')
#    dst = img.convert('1')
#    print(np.array(dst))
    cv2.imwrite('./test/0032.png', dst)

if __name__ == '__main__':
    main()