import os
import cv
import cv2
import numpy as np
import pickle
import matplotlib
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import random
import copy
import math

#Q1
img1 = mpimg.imread('aloeL.png')	
img2 = mpimg.imread('aloeR.png')

blur1 = cv2.GaussianBlur(img1,(15,15),0)  
blur2 = cv2.GaussianBlur(img2,(31,31),0)

imgp = plt.imshow(blur1)
plt.savefig("blur1.jpg")
plt.show()

imgp = plt.imshow(blur2)
plt.savefig("blur2.jpg")
plt.show()

#Q2

blur1a = cv2.imread('blur1.jpg',0)
blur2a = cv2.imread('blur2.jpg',0)

dft1 = cv2.dft(np.float32(blur1a),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift1 = np.fft.fftshift(dft1)
magnitude_spectrum1 = 20*np.log(cv2.magnitude(dft_shift1[:,:,0],dft_shift1[:,:,1]))
plt.imshow(magnitude_spectrum1,cmap = 'gray')
# plt.savefig("FT_1.jpg")
plt.show()


dft = cv2.dft(np.float32(blur2a),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.imshow(magnitude_spectrum,cmap = 'gray')
# plt.savefig("FT_2.jpg")
plt.show()


def calci(i,j,m1,m2):
	x = 2.0/(((i*i)+(j*j))*1.0)
	y = math.log10((m1*1.0)/(m2*1.0))
	return x*y

sigma_relative =[]

for i in range(0,len(magnitude_spectrum)):
	temp = []
	for j in range(0,len(magnitude_spectrum[0])):
		temp.append(calci(i+1,j+1,magnitude_spectrum1[i][j],magnitude_spectrum[i][j]))
	sigma_relative.append(temp)
sigma_relative = np.asarray(sigma_relative)

print(sigma_relative)
