import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.filters import threshold_local
import os


# Load image with cv2 and convert to binary images
# Récupérer les fichiers dans le repertoire
files_in_dir = os.lisdir()
# Récupérer le chemin d'accès courant
curr_path = os.getcwd()


# Get image file names in current directory
image_names = []

# Liste des extensions d'images autorisées
conventions = ['jpeg','png','jpg']

for file in files_in_dir:
    # Déterminer l'extension du fichier image
    ext = file.split('.')[-1]
    if ext in conventions:
        image_names.insert(0,file)

# Read images in numpy array with opencv
images_read = []
for name in image_names:
    img = cv2.imread(name)
    images_read.insert(0,img)

#Convert RGB images to Grayscale
thsh_images = []
for img in images_read:
    # Conversion des images RGB en binary images
    image_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Conversion de la forme RGB en format Gray scale
    clahe = cv2.createCLAHE(clipLimit=4.0,titleGridSize=(16,16))
    image_gray = clahe.apply(image_gray)

    # Ici , il s'agit d'améliorer le contraste des images en appliquant un "Histogram Equalization"
    # On applique un seuil (threshold) au image format Grayscale avec une valeur de thresh de 130 et on stocke ces binary images dans une liste avec Python
    ret,th = cv2.threshold(image_gray,130,255,cv2.THRESH_BINARY)
    thsh_images.append(th)

# Find contours in image and fit Maximum area contour to image
# Find contours in image using (tree retrieval method) for hierachy

image_conts = []
for img in thsh_images:
    _,contours,_ = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    



