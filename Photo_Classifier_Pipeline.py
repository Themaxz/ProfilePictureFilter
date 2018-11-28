
# coding: utf-8

# In[3]:


#Script to classify profile pictures with a pink background (or any other colour)
import cv2
import urllib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os 
import io
get_ipython().magic('matplotlib inline')
import json
#loading configuration file
with open('config.json', 'r') as f:
    config = json.load(f)


# In[4]:


root_dir = config['FOLDERS']['ROOT_DIR']
path = config['FOLDERS']['PATH']
version = config['VERSION']
results_path = config['FOLDERS']['RESULTS']
version = config['VERSION']
members_list = config['FILES']['MEMBERS']
members_messaged = config['FILES']['MEMBERS_MESSAGED']

path =  path+ r'{}'.format(version)
savepath = results_path+ r'{}'.format(version)
def save(df,path,name):
    n = r'\{}'.format(name)
    df.to_csv(path+n, index=False)

def load(path,name):
    n = r'\{}'.format(name)
    return pd.read_csv(path+n,encoding='iso-8859-1')
def GetWeekStart(week,year,weekday='-1'):
    d = str(year)+"-W"+str(week)
    r = dt.datetime.strptime(d + weekday, "%Y-W%W-%w")
    return r.strftime("%d-%b-%Y")

if not os.path.exists(path):
    os.makedirs(path)
    
    
if not os.path.exists(savepath):
    os.makedirs(savepath)


# In[6]:


#all members thumbnails
members = load(path,members_list)


# In[7]:


#messaged members
members_messaged = load(path,members_messaged)


# In[8]:


userIds = members_messaged.UserId
messaged_members_url = members[members['Id'].isin(list(userIds))]
url = list(messaged_members_url.PictureUrl)


# In[10]:


#Download thumbnails
images = []
labels = []
reds = []
members_url
if not os.path.exists(path):
    os.makedirs(path)
i = 0
for index,u in enumerate(messaged_members_url.PictureUrl):
    req = urllib.request.urlopen(u)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1) # 'Load it as it is'

    i+=1
    userId = messaged_members_url[messaged_members_url.PictureUrl==u].Id.values[0]
    print (userId)
    filename = 'img'+str(userId)+'.png'
    cv2.imwrite(os.path.join(path,filename),img)


# In[143]:


#photos to classify
Data = []
Ids = []
dir_name = path
for file in os.listdir(dir_name):
    print(file)
    uid = file.split('.')[0]
    uid = uid.split('mg')[1]
    Ids.append(uid)
    img = cv2.imread(os.path.join(dir_name, file))
    Data.append(img)
    
Photos = pd.DataFrame()
Photos['UserId'] = Ids


# In[144]:


size_x = config['IMAGES']['THUMBNAIL']['SIZE']
size_y = config['IMAGES']['THUMBNAIL']['SIZE']
fil = cv2.imread('WPFilter.png')
fil = cv2.resize(fil,(size_x,size_y))
    


# In[145]:


from sklearn.cluster import KMeans
from collections import Counter
import math

#DATA PREPROCESS
#features engineering
def PreparePictures(Data,Photos,fil):
    
    fil = cv2.cvtColor(fil, cv2.COLOR_BGR2RGB)
    average_color_filter = [fil[:, :, i].mean() for i in range(fil.shape[-1])]   

    R = []
    G = []
    B = []
    dominant = []
    average = []
    dist = []
    dist_av = []
    dist_dom_mean = []
    dom_fil = get_dominant_color(fil)
    for img in Data:
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dom = get_dominant_color(img1)
        print(dom)
        r = dom[0]
        g = dom[1]
        b = dom[2]
        dominant.append(dom)
        R.append(r)
        G.append(g)
        B.append(b)
        dist.append(GetDistance(dom,dom_fil))
        average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]   
        average.append(average_color)
        dist_av.append(GetDistance(average_color,average_color_filter))
        dist_dom_mean.append(GetDistance(average_color,dom))

    Photos['R'] = R
    Photos['G'] = G
    Photos['B'] = B
    Photos['DominatColour'] = dominant
    Photos['Distance'] = dist
    Photos['AverageColour'] = average
    Photos['AverageColourDist'] = dist_av
    Photos['DistAverageDominat'] = dist_dom_mean
    return Photos

def get_clusters(image, k=4, image_processing_size = None):
#reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    #count labels to find most popular
    label_counts = Counter(labels)
    return clt.cluster_centers_, clt

def get_dominant_color(image, k=4, image_processing_size = None):
    """
    takes an image as input
    returns the dominant color of the image as a list
    
    dominant color is found by running k means on the 
    pixels & returning the centroid of the largest cluster
    """
    #resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    
    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    #count labels to find most popular
    label_counts = Counter(labels)

    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)

def GetDistance(vec1,vec2):
    """
    Gets Euclidean Distance Between two colours in RGB space
    """
    diff = np.asarray(vec1) - np.asarray(vec2)
    squareDistance = np.dot(diff.T, diff)
    return math.sqrt(squareDistance)


# In[146]:


Photos = PreparePictures(Data,Photos,fil)


# In[147]:


def RGBratios(dataset,fil):
    """
       Gets the rations between RGB values. This helps identifying the background colour of an image
    """
    dom_fil = get_dominant_color(fil)
    dataset['R_B_ratio'] = dataset.R/dataset.B
    dataset['R_G_ratio'] = dataset.R/dataset.G
    dataset['G_B_ratio'] = dataset.G/dataset.B
    filter_RB = dom_fil[0]/dom_fil[2]
    filter_RG = dom_fil[0]/dom_fil[1]
    filter_GB = dom_fil[1]/dom_fil[2]
    norm = []
    for array in list(dataset['DominatColour']):
        norm.append( np.linalg.norm(array))
    dataset['Dominant_norm'] = norm

    norm = []
    for array in list(dataset['AverageColour']):
        norm.append( np.linalg.norm(array))


    dataset['DiffRatio_R_B'] = dataset.R_B_ratio-filter_RB
    dataset['DiffRatio_R_G'] = dataset.R_G_ratio-filter_RG
    dataset['DiffRatio_RG_B'] = dataset.G_B_ratio-filter_GB
    dataset['Average_norm'] = norm
    return dataset


# In[148]:


buffer = Photos
photos = RGBratios(Photos, fil)


# In[149]:


photos = RGBratios(Photos, fil)


# In[150]:


photos


# In[122]:


data


# In[108]:


dataset = pd.read_csv('ImagesSet.csv')
target = dataset.IsPink
data = dataset.drop(['IsPink'],axis=1)


# DROP VECTORS FEATURES

data.drop(['DominatColour','AverageColour'], axis=1,inplace=True)

#DIVIDE DATASET INTO TRAINING AND TEST
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)


# In[109]:


train = x_train
test = y_test
#DROP USERID
x_train.drop(['UserId'],axis=1,inplace=True)
x_test.drop(['UserId'],axis=1,inplace=True)


# In[110]:


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()


# In[111]:


#TRAIN THE MODEL
logisticRegr.fit(x_train, y_train)


# In[112]:


#PREDICTION ON TEST 
# Returns a NumPy Array
# Predict for One Observation (image)
predictions = logisticRegr.predict(x_test)


# In[113]:


#RESULT DATAFRAME
Predictions = pd.DataFrame()
Predictions['UserId'] = data.ix[x_test.index].UserId
Predictions['Prediction'] = predictions
Predictions['FileName'] = Predictions['UserId'].apply(lambda x: 'img'+str(x)+'.png')


# In[114]:


score = logisticRegr.score(x_test, y_test)
print(score)


# In[115]:


Predictions.groupby(['Prediction'])['UserId'].count()


# In[116]:


Predictions['Expected'] = dataset.ix[x_test.index].IsPink


# In[117]:


Predictions.to_csv('Predictions.csv')


# In[151]:


new_data = photos


# In[152]:


new_data.drop(['DominatColour','AverageColour'], axis=1,inplace=True)
Summary = pd.DataFrame()
Summary['UserId'] = new_data['UserId']
new_data.drop(['UserId'], axis=1,inplace=True)
Newpredictions = logisticRegr.predict(new_data)
Summary['Prediction'] = Newpredictions


# In[153]:


Summary.groupby(['Prediction'])['UserId'].count()


# In[139]:


new_data


# In[155]:


Summary.to_csv('PhotoSummary.csv',index=False)

