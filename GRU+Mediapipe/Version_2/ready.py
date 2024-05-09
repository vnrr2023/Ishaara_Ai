import mediapipe as mp
import cv2 as cv
import cvzone
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import warnings
warnings.filterwarnings("ignore")
import pickle





def extract_landmarks_for_ready(results):
    lh=np.array([[landmark.x,landmark.y,landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh=np.array([[landmark.x,landmark.y,landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    pose=np.array([[landmark.x,landmark.y,landmark.z] for landmark in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    return np.concatenate([lh,rh,pose])


def isReady():

    detector=FaceMeshDetector(maxFaces=1)
    mp_drawing=mp.solutions.drawing_utils
    mp_holistic=mp.solutions.holistic
    holistic=mp_holistic.Holistic(min_detection_confidence=0.7)
    print("loading ready model...")
    ready_model=pickle.load(open('../ready_model/ready_model.pickle','rb'))
    print("ready model loaded successfully...")

    VAL_1=False  #user in range
    VAL_2=False  #user is ready
    COMPARE_1=179-3
    COMPARE_2=192
    d=0

    start_point1=(180+20,0)
    end_point1=(180+20,480)
    start_point2=(440,0)
    end_point2=(440,480)


    cap=cv.VideoCapture(0)
    while True:
        _,frame=cap.read()
        image=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        results=holistic.process(image)

        landmarks=extract_landmarks_for_ready(results)
        
        img,faces=detector.findFaceMesh(frame,draw=False)
        

        cv.line(img,start_point1,end_point1,(0,0,255),3)
        cv.line(img,start_point2,end_point2,(0,0,255),3)
        
        if faces:
            face=faces[0]
            left_eye,right_eye=face[145],face[374]
            cv.line(img,left_eye,right_eye,(0,255,0),3,cv.LINE_AA)
            cv.circle(img,left_eye,3,(0,0,255),cv.FILLED)
            cv.circle(img,right_eye,3,(0,0,255),cv.FILLED)

            dist_in_pixels , _ = detector.findDistance(left_eye,right_eye)
            average_human_eye_dist=6.3  # in cms
            focal_lenght=840
            # finding the d which is distance from object to focal point
            d=int((average_human_eye_dist*focal_lenght)/dist_in_pixels)
            # cv.putText(img,f"{d} in cm",(200,400),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0),2,cv.LINE_AA)
            
        # if user in range of predictions
        if COMPARE_1 <= d <=COMPARE_2:
            VAL_1=True
            cv.line(img,start_point1,end_point1,(0,255,0),3)
            cv.line(img,start_point2,end_point2,(0,255,0),3)
        else:
            VAL_1=False
            VAL_2=False

        if VAL_1:
            text=ready_model.predict(np.expand_dims(landmarks,0))[0]
            cv.putText(img,text,(400,400),cv.FONT_HERSHEY_COMPLEX_SMALL,3,(255,0,0),1,cv.LINE_AA)
            if text=='ready':
                VAL_2=True
                
        if VAL_1 and VAL_2:
            cap.release()
            cv.destroyAllWindows()
            break
        
        if cv.waitKey(1)==27:
            cap.release()
            cv.destroyAllWindows()
            break

        cv.imshow('frame',img)

# \ready\ready_model.pickle
def getting_started():
    cap=cv.VideoCapture(0)
    i=0
    while True:
        _,frame=cap.read()
        
        if 0<i<70:
            cv.putText(frame,'Getting Started',(100,250),cv.FONT_HERSHEY_DUPLEX,2,(0,255,0),2,cv.LINE_AA)
        elif 70<i<140:
            cv.putText(frame,'3',(260,250),cv.FONT_HERSHEY_DUPLEX,4,(0,0,255),2,cv.LINE_AA)
        elif 140<i<190:
            cv.putText(frame,'2',(260,250),cv.FONT_HERSHEY_DUPLEX,4,(0,0,255),2,cv.LINE_AA)
        elif 190<i<240:
            cv.putText(frame,'1',(260,250),cv.FONT_HERSHEY_DUPLEX,4,(0,0,255),2,cv.LINE_AA)
        elif 240<i<300:
            cv.putText(frame,'GO',(260,250),cv.FONT_HERSHEY_DUPLEX,4,(0,255,0),2,cv.LINE_AA)
        i+=1
        if i==300:
            cap.release()
            cv.destroyAllWindows()
            break
        if cv.waitKey(10)==27:
            cap.release()
            cv.destroyAllWindows()
            break

        cv.imshow('frame',frame)