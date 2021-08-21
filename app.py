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
    # cv2.RETR_TREE finds contours in hierachy order
    # cv2.CHAIN_APPROX_SIMPLE just fits outline position co-ordinates of contour which describe
    # entire contour instead entire contour co-ordinates which take more spac
    _,contours,_ = cv2.findContours(img.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    image_conts.append(contours)

# look for maximum area contours which describes page/rectangle structure in image
max_conts_area = []

for contour in image_conts:
    max_ind,max_area = None,0
    for ind,cnt in enumerate(contour):
        # Retrieve area of each contour and finds maximum area of contour to fit
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_ind = ind
    max_conts_area.append(max_ind)

# Fit contours of maximum area
for ind,contour in enumerate(image_conts):
    img = images_read[ind].copy()
    img = cv2.drawContours(img,contour,3,(0,255,0),4)

# Draw closest rectangle shape to maximum contour which usually describes a page of PDF
approx_cont = []
for ind in range(len(images_read)):
    epsilon = 0.02*cv2.arcLength(image_conts[ind][max_conts_area[ind]],True)
    approx = cv2.approxPolyDP(image_conts[ind][max_conts_area],epsilon,True)
    approx_cont.append(np.squeeze(approx))

# Using perspective transformation transform four sides of rectangle to full oc-ordinates of an image
rect_images = []
for ind in range(len(images_read)):
    # top-left,bottom-left,bottom-right,top-right
    tl,bl,br,tr = approx_cont[ind].tolist()
    top_width = np.sqrt((tl[0]+tr[0])**2 + (tl[1]+tr[1])**2)
    bottom_width = np.sqrt((bl[0]+br[0])**2 + (bl[1]+br[1])**2)
    left_height = np.sqrt((tl[0]+bl[0])**2 + (tl[1]+bl[1])**2)
    right_height = np.sqrt((tr[0]+br[0])**2 + (tr[1]+br[1])**2)
    width = int(max(top_width,bottom_width))
    height = int(max(left_height,right_height))

    # order is tl,tr,br,bl
    pres = np.array([tl,tr,br,bl],dtype='float32')
    to = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(pres,to)
    dst = cv2.warpPerspective(images_read[ind].copy(),M,(int(width),int(height)))

    rect_images.append(dst)

# Improve gray scale image contrast using sckit-image and save images
# Digitise image in black and white as a scanned document
digitised_image_names = []
for ind in range(len(rect_images)):
    im_gray = cv2.cvtColor(rect_images[ind].copy(),cv2.COLOR_BGR2GRAY)
    # scikit-image's threshold_local() function increases low contrast image very high contrast
    th =threshold_local(im_gray.copy(),101,offset=10,method="gaussian")
    im_gray = (im_gray > th)
    imgg = Image.fromarray(im_gray)
    size = (images_read[ind].shape[0],images_read[ind].shape[1])
    imgg.resize(size)
    # save high contrast images with name "digitised_ + original name" of image
    name = curr_path+"/digitised_"+image_names[ind].split('.')[0]+'.jpg'
    digitised_image_names.append(name)
    imgg.save(digitised_image_names[ind])

# Convert all digitised images to pdf format
digitised_image = []
for name in digitised_image_names:
    imgg=Image.open(name)
    digitised_image.append(imgg)
    name = curr_path+"/digitised_images"+'.pdf'
    if len(digitised_image) > 1:
        digitised_image[0].save(name,save_all=True,append_images=digitised_image[1:],resolution=100.0)
    else:
        digitised_image[0].save(name)









