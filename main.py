import cv2
import numpy as np
from PIL import Image
from kmeans import KMeans

image = cv2.imread('images\input-image.jpg')

raw = np.float32(image.reshape((-1, 3)))

ks = [1, 2, 3, 4, 5, 6, 8, 10, 14, 16, 20, 25, 40, 50]

for k in ks:
    model = KMeans(k=k)
    model.fit(raw)
    segemented_raw = np.zeros(raw.shape)

    for i, pixel in enumerate(raw):
        segemented_raw[i] = np.int64(model.predict(pixel))

    segemented = segemented_raw.reshape(image.shape)
    segemented = cv2.putText(
        segemented, f"k={k}", (330, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
    cv2.imwrite(f"images\segmented-images\k_{k}.jpg", segemented)
    print(f"k_{k}.jpg outputed")

frames = [Image.open(f"images\segmented-images\k_{i}.jpg")
          for i in ks]
frame_one = frames[0]
frame_one.save("output.gif", format="GIF", append_images=frames,
               save_all=True, duration=750, loop=0)

print('''
-------------------------
        Done!!
-------------------------
        ''')
