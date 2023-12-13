import cv2
import numpy as np
from PIL import Image
from kmeans import KMeans
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Image Segmentation using k-means clustering.")
parser.add_argument('--path', type=str, default='images/input-image.jpg', help='Path to the input image')
args = parser.parse_args()

# Read the image from the provided path
image = cv2.imread(args.path)

raw = np.float32(image.reshape((-1, 3)))

ks = [1, 2, 3, 4, 5, 6, 8, 10, 14, 16, 20, 25, 40, 50]

for k in ks:
    model = KMeans(k=k)
    model.fit(raw)
    segmented_raw = np.zeros(raw.shape)

    for i, pixel in enumerate(raw):
        segmented_raw[i] = np.int64(model.predict(pixel))

    segmented = segmented_raw.reshape(image.shape)
    segmented = cv2.putText(
        segmented, f"k={k}", (330, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
    cv2.imwrite(f"images/segmented-images/k_{k}.jpg", segmented)
    print(f"k_{k}.jpg outputed")

frames = [Image.open(f"images/segmented-images/k_{i}.jpg")
          for i in ks]
frame_one = frames[0]
frame_one.save("images/output.gif", format="GIF", append_images=frames,
               save_all=True, duration=750, loop=0)

print('''
-------------------------
        Done!!
-------------------------
''')
