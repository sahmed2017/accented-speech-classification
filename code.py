##Audio Segmentation

from pydub import AudioSegment 
from pydub.utils import make_chunks
import os

def audio_segmentation(file_name):
    myaudio = AudioSegment.from_wav(file_name, "wav") 
    length = 6000 #milliseconds
    chunks = make_chunks(myaudio,lenght) 
    for i, chunk in enumerate(chunks): 
        chunk_name = 'filepath' + "_{0}{1}.wav".format(i,each_file)  
        chunk.export(chunk_name, format="wav") 

all_file_names = os.listdir('filepath')

for each_file in all_file_names:
    if ('.wav' in each_file):
        audio_segmentation('filepath'+each_file)
        
        
##Full code

import pandas as pd
import numpy as np
import librosa
import librosa.display
from keras import layers
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv("40speakers.csv",index_col='speakerid')
data.head()

data['native_language'].value_counts()

directory_path='filepath'

def feature_engineering(directory_path,data):
    p=0
    df=pd.DataFrame()
    tmp={}
    for index, row in data.iterrows():
        if os.path.isfile(directory_path+row['filename']+'.mp3')==False:
            print('File '+str(row['filename'])+".mp3 doesn't exist")
            data=data.drop([index])
            continue
        tmp['filename']=row['filename']
        tmp['native_language']=row['native_language']
        y, sr=librosa.load(os.path.join(os.path.abspath(directory_path),row['filename']+'.mp3'))
        tmp['rms']=np.mean(librosa.feature.rms(y=y))
        tmp['chroma_stft']=np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        tmp['spec_cent']=np.mean(librosa.feature.spectral_centroid(y=y,sr=sr))
        tmp['spec_bw']=np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        tmp['rolloff']=np.mean(librosa.feature.spectral_rolloff(y=y,sr=sr))
        tmp['zcr']=np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc=librosa.feature.mfcc(y=y, sr=sr)
        i=0
        for e in mfcc: 
            tmp['mfcc'+str(i)]=np.mean(e)
            i+=1
        df=df.append([tmp])
        print(p)
        p+=1
    return df

df.to_csv('audio_features.csv')

complete_data = pd.read_csv('audio_features.csv')
complete_data.head()

le=preprocessing.LabelEncoder()
complete_data['native_language']=le.fit_transform(data_to_fit1['native_language'].astype(str))

complete_data.drop(columns=['filename','Unnamed: 0'])
complete_data.head()

x_train, x_test, y_train, y_test=train_test_split(df.drop(columns=['language']), df['language'],test_size=0.2, random_state=10)

##For scaled values
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

def create_network():
    
    model=Sequential()
    model.add(layers.Dense(150, activation='relu',input_shape=(x_train.shape[1],)))
    #model.add(Dropout(.25))
    model.add(layers.Dense(80, activation='relu'))
    #model.add(Dropout(.25))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # Return compiled network
    return model
  
 neural_network = KerasClassifier(build_fn=create_network, 
                                 epochs=130)

np.mean(cross_val_score(neural_network, x_train, y_train, cv=7))

##For metrics

model=Sequential()
model.add(layers.Dense(150, activation='relu',input_shape=(x_train.shape[1],)))
#model.add(Dropout(.25))
model.add(layers.Dense(80, activation='relu'))
#model.add(Dropout(.25))
model.add(layers.Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
classifier=model.fit(x_train, y_train, epochs=130, validation_data=(x_test,y_test))

model.evaluate(x_test, y_test)

y_predicted = model.predict(x_test)

y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
cm

plt.figure()
sns.heatmap(cm, cmap="Blues", annot=True)
plt.xlabel('Predicted')
plt.title("MFCC & ")
plt.ylabel('True')

##Graphs

print(" :" , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(130)]
fig , ax = plt.subplots(1,2)
train_acc = classifier.history['accuracy']
train_loss = classifier.history['loss']
test_acc = classifier.history['val_accuracy']
test_loss = classifier.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

##Traditional ML Algorithms

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=5),x_train,y_train, cv=7))

knn.score(x_test, y_test)

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)

np.mean(cross_val_score(RandomForestClassifier(n_estimators=1000),x_train,y_train, cv=7))

model = SVC()
model.fit(x_train, y_train)

np.mean(cross_val_score(SVC(),x_train,y_train, cv=7))

#TSNE

from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.offline as py
import plotly.express as px

data=pd.read_csv('audio_features.csv')
data

le=preprocessing.LabelEncoder()
data['language']=le.fit_transform(data['language'].astype(str))
data['language']=le.fit_transform(data['language'].astype(int))

target = data['language']

data = data.drop(columns=['Unnamed: 0','filename','language','rms','spec_bw','zcr','rolloff','chroma_stft','spec_cent'])
data.head()

X = data.values
X_scaled = StandardScaler().fit_transform(X)

model = TSNE(n_components=2, learning_rate=, perplexity=100, verbose=2).fit_transform(X_scaled)



x_axis=model[:,0]
y_axis=model[:,1]
fig = px.scatter(x=x_axis, y=y_axis, color=target,opacity=0.7)
fig.show()


