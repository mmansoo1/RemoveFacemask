import numpy as np
import cv2
import math
import random
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

def skin_rgb_threshold(src):
    # extract color channels and save as SIGNED ints
    # need the extra width to do subraction
    b = src[:,:,0].astype(np.int16) 
    g = src[:,:,1].astype(np.int16)
    r = src[:,:,2].astype(np.int16)

    skin_mask =                                    \
          (r > 96) & (g > 40) & (b > 10)           \
        & ((src.max() - src.min()) > 15)           \
        & (np.abs(r-g) > 15) & (r > g) & (r > b)    

    return src * skin_mask.reshape(skin_mask.shape[0], skin_mask.shape[1], 1)

im1 = cv2.imread('pp1.png')
im2 = cv2.imread('pp2.png')

imagePath = "pp1.png" 
imagePath2 = 'pp2.png'
cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)
faces2 = faceCascade.detectMultiScale(
    gray2,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} face(s)!".format(len(faces)))
print("Found {0} face(s) with a mask!".format(len(faces2)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(im1, (x, y-40), (x+w, y+h+80), (0, 255, 0), 2)
    cut =im1[y-40:y+h+80,x:x+w,:] 
    cv2.imwrite("cut1.png",cut)
    plt.figure()
    plt.title('face with no mask')
    plt.imshow(cv2.cvtColor(im1,cv2.COLOR_BGR2RGB))

for (x, y, w, h) in faces2:
    cv2.rectangle(im2, (x, y-40), (x+w, y+h+80), (0, 255, 0), 2)
    cut =im2[y-40:y+h+80,x:x+w,:] 
    cv2.imwrite("cut2.png",cut)
    plt.figure()
    plt.title('face with mask')
    plt.imshow(cv2.cvtColor(im2,cv2.COLOR_BGR2RGB))

cv2.imshow("Faces found", im1)
cv2.imshow("Faces found (with mask)", im2)

cutImage = cv2.imread('cut1.png')
cutImage2 = cv2.imread('cut2.png')

plt.figure()
plt.title("Before Skin Detection (without mask)")
img = cv2.cvtColor(cutImage, cv2.COLOR_BGR2GRAY)
plt.imshow(img,cmap='gray', vmin=0, vmax=255)

img = cv2.cvtColor(cutImage, cv2.COLOR_BGR2RGB)
skin_detect = skin_rgb_threshold(cutImage)
plt.figure()
plt.title("After Skin Detection (without mask)")
plt.imshow(cv2.cvtColor(skin_detect, cv2.COLOR_BGR2RGB))

plt.figure()
plt.title("Before Skin Detection (with mask)")
img2 = cv2.cvtColor(cutImage2, cv2.COLOR_BGR2GRAY)
plt.imshow(img2, cmap='gray', vmin=0, vmax=255)

img2 = cv2.cvtColor(cutImage2, cv2.COLOR_BGR2RGB)
skin_detect2 = skin_rgb_threshold(cutImage2)
plt.figure()
plt.title("After Skin Detection (with mask)")
plt.imshow(cv2.cvtColor(skin_detect2, cv2.COLOR_BGR2RGB))

skin_detect2_resized = cv2.resize(skin_detect, (skin_detect.shape[1], skin_detect.shape[0]))
alpha = 0.5
beta = (1.0 - alpha)
dst = cv2.addWeighted(skin_detect, alpha, skin_detect2_resized, beta, 0.0)
plt.figure()
plt.title('merged images of skin detected with and without mask')
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

cut1 = cv2.imread('cut1.png')
cut1_resized = cv2.resize(cut1, (dst.shape[1], dst.shape[0]))
dst2 = cv2.addWeighted(dst, alpha, cut1_resized, beta, 0.0)
plt.figure()
plt.title('Final merged images.')
plt.imshow(cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB))

plt.show()