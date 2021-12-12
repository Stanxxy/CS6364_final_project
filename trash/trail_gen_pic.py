import numpy as np
import time
from PIL import Image

start_time = time.time()
duration = 1000
# print(start_time + duration, time.time())
while start_time + duration > time.time():
    # print(start_time)
    img = np.random.randn(400, 400, 3).astype(np.uint8) + 1
    i = Image.fromarray(img)
    i.save("image_test.png")
