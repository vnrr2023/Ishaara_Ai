import cv2 as cv
import os
from tensorflow import keras
import mediapipe as mp
import numpy as np

mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic
holistic=mp_holistic.Holistic()

style1=mp_drawing.DrawingSpec((0,0,0),2,1)
style2=mp_drawing.DrawingSpec((0,0,0),2,1)

def draw_landmarks(img,results):
    mp_drawing.draw_landmarks(img,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,style1,style2)
    mp_drawing.draw_landmarks(img,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,style1,style2)
    mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,style1,style2)

actions=['blank', 'hello', 'how are you', 'i', 'morning', 'sorry', 'thank you']

import tensorflow as tf
model=keras.models.load_model("conv_lstm.h5")

def process_frame(frame):
    frame=cv.resize(frame,(192,192))
    frame=tf.image.rgb_to_grayscale(frame)/255
    return tf.cast(frame,tf.float32)

try:
    frames=[]
    truth=[]
    predicted=[]
    text="dummy"
    probab="40"
    cap=cv.VideoCapture(0)
# video_dir="../data"
# for action in os.listdir(video_dir):
#     print(action)
#     for video in os.listdir(os.path.join(video_dir,action)):
#         video_path=os.path.join(video_dir,action,video)
#         print(video_path)
#         cap=cv.VideoCapture(video_path)
#         for _ in range(30):
#             _,frame=cap.read()
#             white_img=np.full((480,640),255,dtype=np.uint8)
#             array_with_channels = np.expand_dims(white_img, axis=-1)
#             white = np.repeat(array_with_channels,3,axis=-1)
#             frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
#             results=holistic.process(frame)
#             frame=cv.cvtColor(frame,cv.COLOR_RGB2BGR)
#             draw_landmarks(white,results)
#             processed_frame=process_frame(white)
#             frames.append(processed_frame)
#         data=np.expand_dims(tf.convert_to_tensor(frames),0)
#         frames.clear()
#         result1=model.predict(data)[0]
#         text=actions[np.argmax(result1)]
#         truth.append(action)
#         predicted.append(text)
#         cv.destroyAllWindows()
#         cap.release()
        

# print(truth)
# print(predicted)
# print(dict(zip(truth,predicted)))
# import pickle
# pickle.dump(dict(zip(truth,predicted)),open("test.pkl",'wb'))


    while True:
        _,frame=cap.read()
        white_img=np.full((480,640),255,dtype=np.uint8)
        array_with_channels = np.expand_dims(white_img, axis=-1)
        white = np.repeat(array_with_channels,3,axis=-1)
        frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results=holistic.process(frame)
        frame=cv.cvtColor(frame,cv.COLOR_RGB2BGR)
        draw_landmarks(white,results)
        processed_frame=process_frame(white)
        frames.append(processed_frame)

        if len(frames)==30:
            data=np.expand_dims(tf.convert_to_tensor(frames),0)
            frames.clear()
            result1=model.predict(data)[0]
            text=actions[np.argmax(result1)]
            probab=np.round(np.max(result1),2)
            

        cv.putText(frame,f"{text} {probab}",(30,30),cv.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),1,cv.LINE_AA)
        if cv.waitKey(1)==27:
            break
        cv.imshow('frame',frame)
        cv.imshow('white',white)

    cv.destroyAllWindows()
    cap.release()
except:
    pass