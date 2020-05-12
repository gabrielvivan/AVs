import os
import csv
import cv2
import pandas
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

def preprocess(image):
    if type(None) == type(image):
      return image
    sizey = image.shape[0]    
    image = image[70:sizey-20, :]
    return image






