import mediapipe as mp
import numpy as np
import cv2 as cv
import os
from tensorflow import keras
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
import ready

# mediapipe configuration
print("Configuring Mediapipe")
mp_drawing=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic
holistic=mp_holistic.Holistic()
# specifying styles
style1=mp_drawing.DrawingSpec((71, 237, 212),1,1)
style2=mp_drawing.DrawingSpec((67, 73, 247),2,1)

def draw_landmarks(img,results):
    mp_drawing.draw_landmarks(img,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,style1,style2)
    mp_drawing.draw_landmarks(img,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,style1,style2)
    mp_drawing.draw_landmarks(img,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,style1,style2)
    mp_drawing.draw_landmarks(img,results.face_landmarks,mp_holistic.FACEMESH_CONTOURS,style1,style2)

def extract_landmarks(results):
    face=np.array([[landmark.x,landmark.y,landmark.z] for landmark in results.face_landmarks.landmark]).flatten()  if results.face_landmarks else np.zeros(1404)
    pose=np.array([[landmark.x,landmark.y,landmark.z , landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    rh=np.array([[landmark.x,landmark.y,landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    lh=np.array([[landmark.x,landmark.y,landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)

    return np.concatenate([pose,face,lh,rh])


print("loading isl version 2 model ... ")
model_path=os.path.join("models","gru_isl_model_v2.h5")
model=keras.models.load_model(model_path)

print("Finished loading isl version 2 model sucessfully ...")
time.sleep(5)
print()
print("loading isl version 2 actions ... ")
actions=pickle.load(open('v2_classes.pickle','rb'))

print("Finished loading isl version 2 actions sucessfully ...")
time.sleep(5)
print()
print(" \t\t\t\t Actions are => ")
for i in actions:
    print(f"\t\t\t\t\t\t\t{i}")
    time.sleep(2)


ready.isReady()
ready.getting_started()

print("starting predictions...........")

sentence=[]
frames=[]

threshold=0.7  # change acc to detections

white=cv.imread("white.png")

cap=cv.VideoCapture(0)
while cap.isOpened():
    _,frame=cap.read()

    img=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results=holistic.process(img)
    img=cv.cvtColor(img,cv.COLOR_RGB2BGR)
    draw_landmarks(img,results)
    landmarks=extract_landmarks(results)
    # draw_landmarks(image,results)
    # # landmarks=extract_landmarks(results)
    # frames.insert(0,extract_landmarks(results))
    frames.append(landmarks)
    
    if len(frames)==30:
        x=np.array(frames)
        frames.clear()
        # res_1=actions[np.argmax(model.predict(np.expand_dims(x,0))[0])]
        res_1=model.predict(np.expand_dims(x,0))[0]  # this has the array of predictions
        # sentence.append(res)
    
    # arrangement logic of words 
        if np.max(res_1)>=threshold:  # if our predicted value is greater than the threshold
            res_2=actions[np.argmax(res_1)]  #  predicted output will be stored in res_2 eg=> thanks
            if len(sentence)>0:  # if length of sentence array is greater than 0 which means there are some words present.
                if res_2!=sentence[-1]:  # if the latest outcome is not equal to the latest predicted outcome
                    sentence.append(res_2)
                
            else: # if the sentence lenght is 0 which means its the first prediction
                sentence.append(res_2) 
            
     
    cv.putText(white,' '.join(sentence),(10,20),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1,cv.LINE_AA)
    if cv.waitKey(1)==27:
        cap.release()
        cv.destroyAllWindows()
        break

    cv.imshow("frame",img)
    cv.imshow('ur_sentence',white)
