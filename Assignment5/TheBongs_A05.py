#!/usr/bin/env python
# coding: utf-8

# In[43]:


#Q1
from keras import layers, models, optimizers,utils
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.preprocessing import image
from sklearn.preprocessing import StandardScaler,MultiLabelBinarizer,OneHotEncoder,LabelEncoder
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


#Q2

#use these to install the libraries 
#pip3 install python_speech_features
#pip3 install librosa

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import librosa
import matplotlib.pyplot as plt
import librosa.display
import sklearn
import numpy 
import math


# In[5]:


#1.Classify the given set of images using a vanilla CNN( Don’t apply PCA for this!). Sample code for vanilla CNN:


train = pd.read_csv('styles.csv', error_bad_lines=False)    # reading the csv file
train['image'] = train.apply(lambda row: str(row['id']) + ".jpg", axis=1)    #adding a new column to map images
train.head(10)     #column image has the image ids


# In[6]:


train_image = []
y=[]
for i in tqdm(range(train.shape[0])):
    try:
        img = image.load_img('images/'+str(train['image'][i]),target_size=(80,60))    #mapping the images to the images folder
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
        y.append(train['masterCategory'][i])       #taking classification variable from the metadata
    except:
        continue
        
X = np.array(train_image)    


# In[ ]:





# In[7]:


#encoding the classification variable
cat=LabelEncoder()
Y=np.array(y)
new = cat.fit_transform(Y)
y_f=keras.utils.to_categorical(new,num_classes=len(np.unique(y)))
y_f.shape


# In[8]:


X.shape


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(X, y_f, random_state=42, test_size=0.1)


# In[10]:


#Applying CNN
model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(80, 60,3))) 
#input image dimension: 80x 60
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=256, verbose=1, validation_data=(x_test, y_test))


# In[50]:


#2.PCA is one of the most common dimensionality reduction techniques used. 
#Using PCA with number of components ranging from 2 to 5, classify the given set of images using
#a.K-Nearest Neighbours ( consider k=7)
#b.Artificial Neural Network

#PCA for number components=2
new_x=np.reshape(X,(3553520,180))
x = StandardScaler().fit_transform(new_x)
pca = PCA(n_components=2)
pC = pca.fit_transform(x)
pC.shape=(44419,80,2)
x_train_k, x_test_k, y_train_k, y_test_k = train_test_split(pC, y_f, random_state=42, test_size=0.1)
x_train_k.shape
x_test_k.shape
x_train_k.shape=(39977,160)
x_test_k.shape=(4442,160)


# In[51]:


#KNN
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(x_train_k, y_train_k)

y_pred = classifier.predict(x_test_k)
print(classification_report(y_test_k, y_pred))
print("Accuracy score for KNN")
print(accuracy_score(y_test_k, y_pred, normalize=False))


# In[52]:


clf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

clf.fit(x_train_k, y_train_k) 
y_pred_ann = clf.predict(x_test_k)
print(classification_report(y_test_k, y_pred_ann))
print("Accuracy score for ANN ")
print(accuracy_score(y_test_k, y_pred_ann, normalize=False))


# In[53]:


pca3 = PCA(n_components=3)
x.shape=(3553520,180)
pC3 = pca3.fit_transform(x)
pC3.shape=(44419,80,3)
x_train_k3, x_test_k3, y_train_k3, y_test_k3 = train_test_split(pC3, y_f, random_state=42, test_size=0.1)
x_train_k3.shape=(39977,240)
x_test_k3.shape=(4442,240)


#KNN
classifier.fit(x_train_k3, y_train_k3)
y_pred3 = classifier.predict(x_test_k3)
print(classification_report(y_test_k3, y_pred3))
print("Accuracy score for KNN")
print(accuracy_score(y_test_k3, y_pred3, normalize=False))
#ANN
clf.fit(x_train_k3, y_train_k3) 
y_pred_ann3 = clf.predict(x_test_k3)
print(classification_report(y_test_k3, y_pred_ann3))
print("Accuracy score for ANN ")
print(accuracy_score(y_test_k3, y_pred_ann3, normalize=False))


# In[54]:


pca4 = PCA(n_components=4)
#x = StandardScaler().fit_transform(new_x)
x.shape=(3553520,180)
pC4 = pca4.fit_transform(x)
pC4.shape=(44419,80,4)
x_train_k4, x_test_k4, y_train_k4, y_test_k4 = train_test_split(pC4, y_f, random_state=42, test_size=0.1)
x_train_k4.shape=(39977,320)
x_test_k4.shape=(4442,320)


#KNN
classifier.fit(x_train_k4, y_train_k4)
y_pred4 = classifier.predict(x_test_k4)
print(classification_report(y_test_k4, y_pred4))
print("Accuracy score for KNN ")
print(accuracy_score(y_test_k4, y_pred4, normalize=False))

#ANN
clf.fit(x_train_k4, y_train_k4) 
y_pred_ann4 = clf.predict(x_test_k4)
print(classification_report(y_test_k4, y_pred_ann4))
print("Accuracy score for ANN ")
print(accuracy_score(y_test_k4, y_pred_ann4, normalize=False))


# In[55]:


pca5 = PCA(n_components=5)
#x = StandardScaler().fit_transform(new_x)
x.shape=(3553520,180)
pC5 = pca5.fit_transform(x)
pC5.shape=(44419,80,5)
x_train_k5, x_test_k5, y_train_k5, y_test_k5 = train_test_split(pC5, y_f, random_state=42, test_size=0.1)
x_train_k5.shape=(39977,400)
x_test_k5.shape=(4442,400)


#KNN
classifier.fit(x_train_k5, y_train_k5)
y_pred5 = classifier.predict(x_test_k5)
print(classification_report(y_test_k5, y_pred5))
print("Accuracy score for KNN ")
print(accuracy_score(y_test_k5, y_pred5, normalize=False))


#ANN
clf.fit(x_train_k5, y_train_k5) 
y_pred_ann5 = clf.predict(x_test_k5)
print(classification_report(y_test_k5, y_pred_ann5))
print("Accuracy score for ANN ")
print(accuracy_score(y_test_k5, y_pred_ann5, normalize=False))


# In[56]:


#3. CNN seems to be the most accurate among the three models.


# In[59]:


#Amy has come up with a series of exercises to help with Sheldon’s need for closure.
#The dataset Big Bang Theoryhas an audio clip which contains the best scenes from one of the episodes. 
#Use this audio clip to extract the following features and display theirdimension:
#1.MFCC
#2.Zero Crossing rate
#3.Spectral Centroids
#4.Pitch
#5.Root Mean Square for the signalFind out the use of each of the above feature. 
#Using these features, given a problem of content classification(eg. laughter track vs dialog), 
#which algorithm would you use to classify and why?



#use these to install the libraries 
#pip3 install python_speech_features
#pip3 install librosa



#getting the audio file
audio_file = "The Big Bang Theory Season 6 Ep 21 - Best Scenes.wav"


#mfcc
(rate,sig) = wav.read(audio_file)
mfcc_feat = mfcc(sig,rate)
print("MFCC is:",mfcc_feat)
print("Shape of MFCC:",mfcc_feat.shape)


#Zero crossign rate 
x , sr = librosa.load(audio_file)
# Plot the signal:
get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure(figsize=(10, 5))
librosa.display.waveplot(x, sr=sr)
plt.title('Wave Plot')
plt.show()


# In[58]:


zero_crossings = librosa.zero_crossings(x, pad=False)
print("Number of ZeroCrossings in the entire waveform are:",sum(zero_crossings))
print("Shape of ZeroCrossing:",zero_crossings.shape)


#spectral centroids
#spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
get_ipython().run_line_magic('matplotlib', 'notebook')
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
plt.title("Spectral Centroids")
plt.show()
print("Shape of SpectralCentroids:",spectral_centroids.shape)


#Pitch
pitches, magnitudes = librosa.piptrack(x, sr=sr)
print("Shape of Pitch:",pitches.shape)


#RMS VALUE
rms = math.sqrt(numpy.mean(x*x))
print("RMS value is:",rms)


# In[ ]:


#1
#MFCCs:  Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC.
#The mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum nof a sound

#2
#The zero-crossing rate is the rate of sign-changes along a signal, i.e., 
#the rate at which the signal changes from positive to zero to negative or from negative to zero to positive.
#This feature has been used heavily in both speech recognition
#and music information retrieval, being a key feature to classify percussive sounds.


#3
#The spectral centroid is a measure used in digital signal processing to characterise a spectrum. 
#It indicates where the center of mass of the spectrum is located.

#4
#Pitch is a perceptual property of sounds that allows their ordering on a frequency-related scale,
#t is the quality that makes it possible to judge sounds as "higher" and "lower"  in the sense associated with
#musical melodies.

#5
#RMS is the root-mean-square value of a signal.
#It represents the average "power" of a signal.


#I would use Gaussian Mixture Models for content classification.

