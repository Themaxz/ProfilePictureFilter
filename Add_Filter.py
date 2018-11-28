
# coding: utf-8

# In[3]:


get_ipython().system('jupyter nbconvert --to script Add_Filter.ipynb')
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os 
from PIL import Image
get_ipython().magic('matplotlib inline')

import base64
import datetime
import time
from azure.storage import CloudStorageAccount, AccessPolicy
from azure.storage.blob import BlockBlobService, PageBlobService, AppendBlobService
from azure.storage.models import CorsRule, Logging, Metrics, RetentionPolicy, ResourceTypes, AccountPermissions
from azure.storage.blob.models import BlobBlock, ContainerPermissions, ContentSettings


# In[4]:


import os

import json
with open('config.json', 'r') as f:
    config = json.load(f)

root_dir = config['FOLDERS']['ROOT_DIR']
photos_dir = config['FOLDERS']['PATH']
version = config['VERSION']
filters_dir = config['FOLDERS']['FILTERS']
filters_dir = r'C:\Users\thema\Documents\Ralph Lauren\Filters'
photos_dir = r'C:\Users\thema\Documents\Ralph Lauren\Original'
save_dir = r'C:\Users\thema\Documents\Ralph Lauren\Filtered'


# In[5]:


def save(img,path,name):
    n = r'\{}'.format(name)
    cv2.imwrite(path+n,img)

def load(path,name):
    n = r'\{}'.format(name)
    return cv2.imread(path+n)

def deleteFile(path,name):
    n = r'\{}'.format(name)
    if os.path.exists(path+n):
        print('Deleting file ' +path+n)
        os.remove(path+n)

def saveToBlob(block_blob_service, local_path,local_file_name,container_name):
    # Create a file in Documents to test the upload and download.
    full_path_to_file =os.path.join(local_path, local_file_name)
    print("Temp file = " + full_path_to_file)
    print("\nUploading to Blob storage as blob: " + local_file_name)
    # Upload the created file, use local_file_name for the blob name
    block_blob_service.create_blob_from_path(container_name, local_file_name, full_path_to_file, content_settings=ContentSettings(content_type='image/png'))

def AddFilter(image,background):
    img1 = image
    if img is not None:
    img2 = background
    
    # I want to put logo on top-left corner, So I create a ROI

    rows,cols,channels = img2.shape
    x,y,c = img1.shape

    #ratio = x/cols
    #dst = cv2.resize(img2, (np.int(ratio*cols),np.int(ratio*rows)), interpolation = cv2.INTER_CUBIC)
    #rows,cols,channels = dst.shape
    roi = img1[(y-rows):y,np.int((x/2)-(cols/2)):np.int((x/2)-(cols/2))+cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[(y-rows):y,np.int((x/2)-(cols/2)):np.int((x/2)-(cols/2))+cols] = dst



# In[4]:


#Blob container config
account_key = config['AZURE']['BLOBSTORAGE']['ACCOUNT_KEY']
account_name = config['AZURE']['BLOBSTORAGE']['ACCOUNT_NAME']
container_name = config['AZURE']['BLOBSTORAGE']['ACCOUNT_NAME']
block_blob_service = BlockBlobService(account_name= account_name, account_key=account_key) 

#loading filters
foreground = Image.open(filters_dir+r'/WPFilter.png')
dim = [foreground.height,foreground.width]

#download all blobs and add filter
generator = block_blob_service.list_blobs(container_name)

for blob in generator:
    print("\t Blob name: " + blob.name)
    n = r'\{}'.format( blob.name)
    full_path_to_file2 = os.path.join(photos_dir+ n)
    print("\nDownloading blob to " + full_path_to_file2)
    #download one picture
    block_blob_service.get_blob_to_path(container_name, blob.name, full_path_to_file2)
    #Apply filters
    filename = r""+blob.name
    print("Applying filter to file = " + filename)
    print(filename)
    #load picture to modify
    background = Image.open(photos_dir+ r'\{}'.format(filename))
    print(photos_dir)
    if background is not None:
        background_r = background.resize(dim)
        background_r.paste(foreground, (0, 0),foreground )
        local_file_name = "Filter"+blob.name
        local_file_name = local_file_name.split('.')[0]
        local_file_name = local_file_name +".png"
        full_path_to_file = os.path.join(save_dir+r'\{}'.format(local_file_name))
        background_r.save(full_path_to_file,"PNG")
        #upload to blob storage
       
        print("Temp file = " + full_path_to_file)
        print("\nUploading to Blob storage file:  " + local_file_name)
        ##save to blob modified picture
        saveToBlob(block_blob_service,save_dir,local_file_name,container_name)
        #deleting local files
        deleteFile(save_dir,local_file_name)
        deleteFile(photos_dir,filename)
        
    

