import pandas as pd
import cv2
import zipfile
import numpy as np

tr_csv = pd.read_csv('data/training.csv')
#print(tr_csv.head())


data = zipfile.ZipFile("data/images.zip", 'r')
img = data.read('images/1458171560720DSC_0454.png')
#print(data.namelist())
img = cv2.imdecode(np.frombuffer(img, np.uint8), 1)

class getdata():
    def __init__(self, image_names, labels, file_path):
        assrt len(image_names) == len(labels)
        self.num_examples = len(image_names)
        self.names = image_names
        self.data = {image_names[i]:labels[i] for i in range(len(self.num_examples))}
        self.epoch = 0
        self.startofepoch = 0
        self.path = file_path
        
    def nextbatch(self, batch_size):
        start = self.startofepoch
        end = start + batch_size
        
        if end > self.num_examples:
            self.epoch += 1
            
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            
            self.images = self.images[perm]
            
            start = 0
            end = batch_size
            
        self.startofepoch = end
        return self.get_img_batch(self.images[start:end]), [self.data[i] for i in self.images[start:end]]

    def get_image(self, data, path):
    	img = data.read(path)
    	img = cv2.imdecode(np.frombuffer(img, np.uint8), 1)
    	return img

    def get_img_batch(self, l):
    	return [self.get_image(i) for i in l]

