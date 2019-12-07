import pandas as pd
import cv2
import zipfile
import numpy as np
import os
from tqdm import tqdm
from skimage.feature import hog
from skimage.io import imread
import matplotlib.pyplot as plt

file_name = os.listdir('data/images')
# print(file_name[0])

# data = zipfile.ZipFile("data/images.zip", 'r')
# img = data.read('images/1458171560720DSC_0454.png')
# #print(data.namelist())
# img = cv2.imdecode(np.frombuffer(img, np.uint8), 1)

def get_data(path):
	img = data.read(path)
	
	img = cv2.imdecode(np.frombuffer(img, np.uint8), 1)
	return img

def canny(img, t1, t2):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edge = cv2.Canny(gray, t1, t2)
	return edge

# for i in tqdm(file_name):
	# try:
	# 	edge = canny(get_data('images/'+i), 30, 30)
	# 	cv2.imwrite('data/edge_data/'+i, edge)
	# except:
	# 	print(i)

def hog_tr(frame):
	features, hog_img = hog(frame, 
	                        orientations = 11, 
	                        pixels_per_cell = (8, 8),
	                        cells_per_block = (2, 2), 
	                        transform_sqrt = True, 
	                        visualize = True, 
	                        feature_vector = True)
	return hog_img
  
def hog_transform(frame):
	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
	hogA = hog_tr(frame[:,:,0])
	hogB = hog_tr(frame[:,:,1])
	hogC = hog_tr(frame[:,:,2])
	hog = np.dstack([hogA, hogB, hogC])

	return hog


img = cv2.imread('data/images/1458172053188DSC_0464.png')

hog_fea = hog_transform(img)

cv2.imshow('hog', hog_fea)
cv2.waitKey(0)

cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# img = imread('data/images/1458172053188DSC_0464.png')

# hog_fea = hog_transform(img)

# plt.imshow(img)
# plt.show()
