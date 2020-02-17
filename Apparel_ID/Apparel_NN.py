# importing the libraries
%matplotlib inline
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading dataset
train = pd.read_csv('train_LbELtWX/train.csv')
test = pd.read_csv('test_ScVgIM0/test.csv')

sample_submission = pd.read_csv('sample_submission_I5njJSF.csv')

train.head()


# random number generator
seed = 128
rng = np.random.RandomState(seed)

# print an image
img_name = rng.choice(train['id'])

filepath = 'train_LbELtWX/train/' + str(img_name) + '.png'

img = imread(filepath, as_gray=True)
img = img.astype('float32')

plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')