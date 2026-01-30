import cv2
import torch
import fastcv
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread("artifacts/grayscale.jpg", cv2.IMREAD_GRAYSCALE)

img_tensor = torch.from_numpy(img).cuda()


hist = fastcv.calcHist(img_tensor)


hist_cpu = np.array(hist)


plt.figure(figsize=(10, 6))
plt.bar(range(256), hist_cpu, width=1.0, color='blue', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Image Histogram')
plt.xlim([0, 255])
plt.savefig('output_histogram.png')
plt.close()


print("saved histogram visualization.")

