from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
from sklearn import svm
import argparse
import imutils
import pickle 
import time
import dlib
import cv2

def e_a_r(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    H1 = dist.euclidean(eye[1], eye[5])
    H2 = dist.euclidean(eye[2], eye[4])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    w1 = dist.euclidean(eye[0], eye[3])
 
    # compute the eye aspect ratio
    ear = (H1 + H2) / (2.0 * w1)
 
    # return the eye aspect ratio
    return ear

def passrd(usrnm):
    pswd = ""
    nt1 = usrnm+'_t1_SVM.pickle' 
    nt2 =  usrnm+'_t2_SVM.pickle'
    clf = pickle.load(open(nt1,'rb'))
    clfs = pickle.load(open(nt2,'rb'))
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.19
    EYE_AR_CONSEC_FRAMES_MIN = 3
    EYE_AR_CONSEC_FRAMES_MAX = 24
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    COUNTER_S = 0
    TOTAL_S = 0
    TOTAL_F = 0
    p_f = []
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
 
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    print("Starting video stream thread for password detection...")

    vs = VideoStream(src=0).start()


    time.sleep(1.0)
    clsd = 0
    frameno = 0
    blink_data = []
    start = time.time()

    # loop over frames from the video stream
    while True:
    
        
    
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
    
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = e_a_r(leftEye)
            rightEAR =e_a_r(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            blink_data.append([frameno, ear])
            frameno +=1

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if frameno > 7:
                p_f.append(frameno)
                
                if frameno > 14:
                    curf = p_f[0]
                    p_f = p_f[1:]
                    avg = 0
                    othear = []
                    for i in range(curf-7,curf+7,1):
                        avg += blink_data[i][1]
                        
                        othear.append(blink_data[i][1])
                    avg = avg/15.0
                    ear = blink_data[curf][1]

                    res1 = clf.predict([[ear,othear[0],othear[1],othear[2],othear[3],othear[4],othear[5],othear[6],othear[13],othear[12],othear[11],othear[10],othear[9],othear[8],othear[7],avg]])
                    res2 = clfs.predict([[ear,othear[0],othear[1],othear[2],othear[3],othear[4],othear[5],othear[6],othear[13],othear[12],othear[11],othear[10],othear[9],othear[8],othear[7],avg]])

                    if(res1[0] == 1 ):
                        COUNTER += 1
                    elif(res2[0] == 1 ):
                        COUNTER_S += 1
                    elif(res1[0] == 1 and res2[0] == 1):
                        print("\n\nAWward!!\n\n")
                    else:
                       # if res1[0] == 0:
                            if COUNTER > 0 :
                                if COUNTER > EYE_AR_CONSEC_FRAMES_MIN and COUNTER <= EYE_AR_CONSEC_FRAMES_MAX:
                                    TOTAL_F += 1
                                    frm = curf - COUNTER
                                    print("Blink type1 detected at {} no. frames: {}".format(frm,COUNTER))
                                    print("\n curf:",curf)
                                    pswd = pswd + "1"
                                COUNTER = 0
                       
                       # elif res2[0] == 0:
                            elif COUNTER_S > 0 :
                                if COUNTER_S > EYE_AR_CONSEC_FRAMES_MIN and COUNTER_S <= EYE_AR_CONSEC_FRAMES_MAX:
                                    TOTAL_S += 1
                                    frm = curf - COUNTER_S
                                    print("Blink type2 detected at {} no. frames: {}".format(frm,COUNTER_S))
                                    print("\n curf:",curf)
                                    pswd = pswd + "2"
                                COUNTER_S = 0

            cv2.putText(frame, "Blinks T1: {}".format(TOTAL_F), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            cv2.putText(frame, "Blinks T2: {}".format(TOTAL_S), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, " Press 'q' to exit ", (100, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            #pickle.dump(blink_data, open ("clse.pickle", "wb"))
            end = time.time()
            print("\n Total frames captured :", frameno-1," in:",end - start ," seconds\n")
            break
        if key == ord("r"):
            print("\n\nResetting pasword\n\n")
            time.sleep(1.0)
            COUNTER = 0
            COUNTER_S = 0
            TOTAL_S = 0
            TOTAL_F = 0    
            pswd = "" 

    
    # do a bit of cleanup

    cv2.destroyAllWindows()
    vs.stop()

    return pswd


