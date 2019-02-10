from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import time
import winsound
import threading

cam=cv2.VideoCapture(0)
eye_thresh=0.2
mouth_thresh=0.74
counter=0
consec_frames=9 
alarmRing=False
total=0
consec_yawn_frames=8 #mouth duration
yawn=0
counter_yawn=0

def alert():
     #cv2.putText(frame, "ALERT (Alarm ringing) ", (10, 300),cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
     winsound.Beep(1000, 1000)


def check_yawn(mouth):
    global counter_yawn,mouth_thresh,yawn
    A=distance.euclidean(mouth[0],mouth[6]) #mouth[0] is a list of (x,y) point
    C=distance.euclidean(mouth[2],mouth[10])
    D=distance.euclidean(mouth[4],mouth[8])
    mar=A/(C+D)
    if mar<mouth_thresh:                    #this ratio should be greater than 2.1 for a person who is not yawning
        counter_yawn=counter_yawn+1
    else:
        if counter_yawn>=consec_yawn_frames:
            yawn+=1
            counter_yawn=0
            t1=threading.Thread(target=alert,args=())
            t1.daemon=True
            t1.start()
            

def check_eye():
        global counter,ear,eye_thresh,total
        if ear < eye_thresh:
            counter =counter+1
            
        else:
            if counter >= consec_frames:
                total += 1
                counter = 0 #back to zero
                t2=threading.Thread(target=alert,args=())   #NEVER PUT PARENTHESIS WITH THE FUNCTION NAME, OTHERWISE ALERT WILL BE CALLED
                t2.daemon=True                                #INSIDE THIS CODE BLOCK WITHOUT MULTITHREADING
                t2.start()

        cv2.putText(frame, "Blinks: {}".format(total), (10, 30),cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)

def eye_aspect_ratio(eye):
    v1 = distance.euclidean(eye[1], eye[5])  #Actually, we have sliced the points into this new eye array, and eye[1],eye[5] are 37 and 41 points in the dlib face predictor library
    v2 = distance.euclidean(eye[2], eye[4])
    h = distance.euclidean(eye[0], eye[3])#horizontal
    ear = (v1 + v2) / (2.0 * h)
    return ear

#Read the corresponding points from the face detector which are marked with numbers
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('E:/CSE/CV/project/shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = (42,48)	#points corresponding to eye and mouth
(rStart, rEnd) = (36,42)
(mouthStart,mouthEnd)=(48,67)

while True: 
    ret,frame=cam.read()    #returns tuple
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]   #slicing the required array of points from the whole face
        rightEye = shape[rStart:rEnd]
        mouth=shape[mouthStart:mouthEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull=cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        check_eye()
        check_yawn(mouth)

    cv2.imshow("Driver", frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break    

    
    
cam.release()    
cv2.destroyAllWindows()


