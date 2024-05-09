from time import time
st=time()
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
import os
import cv2 as cv
from tensorflow import keras


video_dir='../processed_videos'
actions=os.listdir(video_dir)


print(actions)

def load_per_video(path):
    cap=cv.VideoCapture(path)
    frames=[]
    for _ in range(int(cap.get(cv.CAP_PROP_FRAME_COUNT))):
        _,frame=cap.read()
        frame=cv.resize(frame,(192,192))
        frame=tf.image.rgb_to_grayscale(frame)
        frames.append(frame)
    cap.release()

    return tf.convert_to_tensor(frames,tf.float32)/255

def load_data(path):
    path=path.numpy().decode("utf-8")
    action=path.split("\\")[2]
    label=actions.index(action)
    # array=np.zeros(7,np.int32)
    # array[label]=1
    tensor=load_per_video(path)
    return tensor,tf.convert_to_tensor(label)

dataset=tf.data.Dataset.list_files(video_dir+"/*/*")
print(len(dataset))


dataset=dataset.shuffle(500)
dataset = dataset.map(lambda x: tf.py_function(load_data, [x], [tf.float32, tf.int32]))
dataset = dataset.padded_batch(6, padded_shapes=([30,192,192,1],[]))
dataset = dataset.prefetch(tf.data.AUTOTUNE)

data,label=dataset.as_numpy_iterator().next()
print(data.shape,label.shape)
del data
del label


train=dataset.take(
    int(
        len(dataset)*0.9
    )
)

test=dataset.skip(
    int(
        len(dataset)*0.9
    )
)




# Define the filepath for saving the model
checkpoint_filepath = 'best_model.h5'

# Define the ModelCheckpoint callback
checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                             monitor='val_loss',  # Monitor validation loss
                             save_best_only=True,  # Save only the best model
                             mode='min',  # Save when the monitored quantity (val_loss) decreases
                            verbose=1)  # Print messages

print("Model part came")
model=keras.Sequential()
model=keras.models.load_model("raw_model.h5")
print("model loaded")

model.add(keras.layers.ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation='tanh', data_format="channels_last",
                         return_sequences=True, input_shape = (30,192, 192, 1)))
model.add(keras.layers.MaxPooling3D(  data_format="channels_last" ))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.3)))

model.add(keras.layers.ConvLSTM2D(filters = 8, kernel_size = (3, 3),activation='tanh', data_format="channels_last",
                        return_sequences=True))
model.add(keras.layers.MaxPooling3D( data_format="channels_last", ))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.3)))

model.add(keras.layers.ConvLSTM2D(filters = 16, kernel_size = (3, 3),activation='tanh', data_format="channels_last",
                         return_sequences=True))

model.add(keras.layers.MaxPooling3D( data_format="channels_last", ))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.3)))

model.add(keras.layers.ConvLSTM2D(filters = 32, kernel_size = (3, 3), activation='tanh', data_format="channels_last",
                         return_sequences=True))

model.add(keras.layers.MaxPooling3D(  data_format="channels_last", ))
model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.3)))

model.add(keras.layers.Flatten()) 

model.add(keras.layers.Dense(len(actions), activation = "softmax"))
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print("\n\nTraining Started \n\n")

history=model.fit(train,epochs=40,validation_data=test,callbacks=[checkpoint])

print(model.evaluate(test))

model.save("conv3d_gru_model.keras")
model.save("conv3d_gru_model.h5")

print("\n\nModel saved\n\n")

import pickle

pickle.dump(history.history,open('history.pkl','wb'))
print("\n\n history saved\n\n")
print(time()-st)