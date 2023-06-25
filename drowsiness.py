
import time
import cv2
import mediapipe as mp
from mediapipe import solutions
import numpy as np
import math
from pygame import mixer

mixer.init()
mixer.music.load("beep.wav")

def dist(p,q):
    return math.sqrt(sum([(px-qx)**2 for px ,qx in zip(p,q)]))


def ear(a1,a2,a3,a4,a5,a6):
    A= dist(a2,a6)
    B= dist(a3,a5)
    C= dist(a1,a4)
    # A= np.linalg.norm(a2,a6)
    # B= np.linalg.norm(a3,a5)
    # C= np.linalg.norm(a1,a4)
    return ((A+B)/(2*C))

def mar(a1,a2,a3,a4,a5,a6,a7,a8):
    A= dist(a2,a8)
    B= dist(a3,a7)
    C= dist(a4,a6)
    D= dist(a1,a5)
    return ((A+B+C)/(2*D))

yawn_alert=0
sleep_alert=0
distraction_alert=0
e_counter=0
y_counter=0
distraction=0


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# cap.set(3,1920)
# cap.set(4,1080)
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Set the timer parameters
text_timer = 0
text_duration = 5  # seconds
# Set the timer parameters
text_timer1 = 0
text_duration1 = 5  # seconds
while True:
    success, image = cap.read()
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    # image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    # image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                # if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx==133 or idx==362:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        # nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    
                    if idx == 33:
                        x1 = (lm.x * img_w, lm.y * img_h)
                    if idx == 160:
                        x2 = (lm.x * img_w, lm.y * img_h)
                    if idx == 158:
                        x3 = (lm.x * img_w, lm.y * img_h)
                    if idx == 133:
                        x4 = (lm.x * img_w, lm.y * img_h)
                    if idx == 153:
                        x5 = (lm.x * img_w, lm.y * img_h)
                    if idx == 144:
                        x6 = (lm.x * img_w, lm.y * img_h)
                    

                    if idx == 362:
                        y1 = (lm.x * img_w, lm.y * img_h)
                    if idx == 385:
                        y2 = (lm.x * img_w, lm.y * img_h)
                    if idx == 387:
                        y3 = (lm.x * img_w, lm.y * img_h)
                    if idx == 263:
                        y4 = (lm.x * img_w, lm.y * img_h)
                    if idx == 373:
                        y5 = (lm.x * img_w, lm.y * img_h)
                    if idx == 380:
                        y6 = (lm.x * img_w, lm.y * img_h)

                    if idx == 61:
                        m1 = (lm.x * img_w, lm.y * img_h)
                    if idx == 73:
                        m2 = (lm.x * img_w, lm.y * img_h)
                    if idx == 11:
                        m3 = (lm.x * img_w, lm.y * img_h)
                    if idx == 303:
                        m4 = (lm.x * img_w, lm.y * img_h)
                    if idx == 291:
                        m5 = (lm.x * img_w, lm.y * img_h)
                    if idx == 404:
                        m6 = (lm.x * img_w, lm.y * img_h)
                    if idx == 16:
                        m7 = (lm.x * img_w, lm.y * img_h)
                    if idx == 180:
                        m8 = (lm.x * img_w, lm.y * img_h)

                    

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
                distraction+=1
            elif y > 10:
                text = "Looking Right"
                distraction+=1
            elif x < -10:
                text = "Looking Down"
                distraction+=1
            elif x > 10:
                text = "Looking Up"
                distraction+=1
            else:
                text = "Forward"
                distraction=0

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            cv2.circle(image, p1, radius=1, color=(255, 0, 0), thickness=5)

            # Display the right eye direction
            x1 = (int(x1[0]), int(x1[1]))
            cv2.circle(image, x1, radius=1, color=(0, 0, 255), thickness=1)
            x2 = (int(x2[0]), int(x2[1]))
            cv2.circle(image, x2, radius=1, color=(0, 0, 255), thickness=1)
            x3 = (int(x3[0]), int(x3[1]))
            cv2.circle(image, x3, radius=1, color=(0, 0, 255), thickness=1)
            x4 = (int(x4[0]), int(x4[1]))
            cv2.circle(image, x4, radius=1, color=(0, 0, 255), thickness=1)
            x5 = (int(x5[0]), int(x5[1]))
            cv2.circle(image, x5, radius=1, color=(0, 0, 255), thickness=1)
            x6 = (int(x6[0]), int(x6[1]))
            cv2.circle(image, x6, radius=1, color=(0, 0, 255), thickness=1)

            # Display the left eye direction
            y1 = (int(y1[0]), int(y1[1]))
            cv2.circle(image, y1, radius=1, color=(0, 0, 255), thickness=1)
            y2 = (int(y2[0]), int(y2[1]))
            cv2.circle(image, y2, radius=1, color=(0, 0, 255), thickness=1)
            y3 = (int(y3[0]), int(y3[1]))
            cv2.circle(image, y3, radius=1, color=(0, 0, 255), thickness=1)
            y4 = (int(y4[0]), int(y4[1]))
            cv2.circle(image, y4, radius=1, color=(0, 0, 255), thickness=1)
            y5 = (int(y5[0]), int(y5[1]))
            cv2.circle(image, y5, radius=1, color=(0, 0, 255), thickness=1)
            y6 = (int(y6[0]), int(y6[1]))
            cv2.circle(image, y6, radius=1, color=(0, 0, 255), thickness=1)

            # Display the mouth direction
            m1 = (int(m1[0]), int(m1[1]))
            cv2.circle(image, m1, radius=1, color=(0, 0, 255), thickness=1)
            m2 = (int(m2[0]), int(m2[1]))
            cv2.circle(image, m2, radius=1, color=(0, 0, 255), thickness=1)
            m3 = (int(m3[0]), int(m3[1]))
            cv2.circle(image, m3, radius=1, color=(0, 0, 255), thickness=1)
            m4 = (int(m4[0]), int(m4[1]))
            cv2.circle(image, m4, radius=1, color=(0, 0, 255), thickness=1)
            m5 = (int(m5[0]), int(m5[1]))
            cv2.circle(image, m5, radius=1, color=(0, 0, 255), thickness=1)
            m6 = (int(m6[0]), int(m6[1]))
            cv2.circle(image, m6, radius=1, color=(0, 0, 255), thickness=1)
            m7 = (int(m7[0]), int(m7[1]))
            cv2.circle(image, m7, radius=1, color=(0, 0, 255), thickness=1)
            m8 = (int(m8[0]), int(m8[1]))
            cv2.circle(image, m8, radius=1, color=(0, 0, 255), thickness=1)


            ear_v=(ear(x1,x2,x3,x4,x5,x6)+ear(x1,x2,x3,x4,x5,x6))/2
            if(ear_v < 0.25):
                e_counter+=1 
            else:
                e_counter=0       
            if(e_counter > 50):
                cv2.putText(image, "SLEEP ALERT", (100,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                mixer.music.play()
                sleep_alert+=1
                e_counter=0

            mar_v=mar(m1,m2,m3,m4,m5,m6,m7,m8)
            if(mar_v > 0.55):
                y_counter+=1 
            else:
                y_counter=0
            if(y_counter >40): 
                cv2.putText(image, "YAWN ALERT", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                mixer.music.play()
                yawn_alert+=1
                y_counter=0

            if(distraction>100):
                cv2.putText(image, "DISTRACTION ALERT", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                mixer.music.play()
                distraction_alert+=1
                distraction=0


            
            # Add the text on the image

            #cv2.putText(image, "EAR : {0:.2f}".format((ear_v)), (50,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            #cv2.putText(image, "MAR : {0:.2f}".format((mar_v)), (50,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(image, f"SLEEP WARNING : {sleep_alert}", (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.putText(image, f"YAWN WARNING : {yawn_alert}", (20,425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.putText(image, f"DISTRACTION WARNING : {distraction_alert}", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if text_timer < text_duration:
                if(yawn_alert>=5 or sleep_alert>=5):
                        cv2.putText(image, " TAKE SOME REST ", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        text_timer += 1/fps
            else:
                text_timer = 0 
                yawn_alert = 0
                sleep_alert = 0  
        # Drawing mesh
        # mp_drawing.draw_landmarks(
        #             image=image,
        #             landmark_list=face_landmarks,
        #             connections=mp_face_mesh.FACEMESH_CONTOURS,
        #             landmark_drawing_spec=drawing_spec,
        #             connection_drawing_spec=drawing_spec)
    else:
        cv2.putText(image, "              MAJOR DISTRACTION                ", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        mixer.music.play()
        


    cv2.imshow('Head Pose Estimation', image)
    if cv2.waitKey(1) & 0xFF ==  ord('q'):
        break


cap.release()