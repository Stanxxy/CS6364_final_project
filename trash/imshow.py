import sys
import cv2
import os
# from PIL import Image
# import numpy as np
# import imghdr
# FILE_NAME = "image.png"
FILE_NAME = 'image_test.png'

# ImageFile.LOAD_TRUNCATED_IMAGES = True


class DevNull:
    def write(self, msg):
        pass


sys.stderr = DevNull()

while True:
    if os.path.exists(FILE_NAME):
        # if imghdr.what('image.png') == "png":
        #     image = np.array(Image.open('image.png').convert("RGB"))
        image = cv2.imread(FILE_NAME, cv2.COLOR_RGB2BGR)
        if not image is None and len(image) > 0:
            cv2.imshow("Frame", image)
            cv2.waitKey(10)
