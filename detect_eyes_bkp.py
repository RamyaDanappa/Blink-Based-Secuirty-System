from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from trainer import svmtrnr

import numpy as np
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



def calibration(usnm,mode):

    # define three constants, one for the eye aspect ratio to indicate
    # blink and then a set of constant for the number of consecutive upper limit and lower limit
    # frames the eye must be below the threshold
    
    name = usnm +"_"+mode

    EYE_AR_THRESH = 0.19
    EYE_AR_CONSEC_FRAMES_MIN = 3
    EYE_AR_CONSEC_FRAMES_MAX = 32
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    print("[INFO] starting video stream thread...")

    vs = VideoStream(src=0)
    vs.start()


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
            #blink_data.append([frameno, ear])
            #frameno +=1

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                blink_data.append([frameno, ear, leftEAR, rightEAR, 0])
                frameno += 1
                
    
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                
                blink_data.append([frameno, ear, leftEAR, rightEAR, 0])
                if TOTAL == 0:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES_MIN and COUNTER <= EYE_AR_CONSEC_FRAMES_MAX:
                        TOTAL = 1
                        
                        for i in range(frameno-COUNTER, frameno,1):
                            blink_data[i][-1] = 1
                        print("\n bLINK AT :", (frameno - COUNTER), " for:",COUNTER ," frames\n" )
                        #initialize a counter forinter blink rate afterdetecting first blink
                        IBT = 0


                elif TOTAL > 0 :
                    
                    if COUNTER >= EYE_AR_CONSEC_FRAMES_MIN and COUNTER <= EYE_AR_CONSEC_FRAMES_MAX and IBT >= 2:
                        TOTAL += 1
                        
                        for i in range(frameno-COUNTER, frameno,1):
                            blink_data[i][-1] = 1
                        print("\n bLINK AT :", (frameno - COUNTER), " for:",COUNTER ," frames\n" )
                    elif COUNTER <  EYE_AR_CONSEC_FRAMES_MIN and COUNTER > 0 :
                        clsd = COUNTER
                        frm = (frameno) - COUNTER
                        blink_data[frameno-COUNTER][-1] = 0 
                        print( " Not blink: {} , from frame: {}".format(clsd, frm) )
                    elif COUNTER == 0:
                        IBT += 1
                    else:
                        clsd = COUNTER
                        frm = (frameno ) - COUNTER
                        for i in range( (frameno - COUNTER), frameno, 1):
                            blink_data[(frameno)-i][-1] = 0
                        print( " Not blink: {} , from frame: {}".format(clsd, frm) )

                frameno += 1
                # reset the eye frame counter
                COUNTER = 0
            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if mode == 't1':
                cv2.putText(frame, " Blink type 1 Press 'q' after blinking", (1, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
            elif mode == 't2':
                 cv2.putText(frame, " Blink type 2 Press 'q' after blinking ", (1, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
               
            cv2.putText(frame, "  sufficient number of times (at least 15) ", (1, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)

        
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        
        if key == ord("q"):
            name = name + '.pickle'
            pickle.dump(blink_data, open (name, "wb"))
            end = time.time()
        #print("\n Total frames captured :", frameno-1," in:",end - start ," seconds\n")
            break
    
    # do a bit of cleanup
    vs.stop()

    cv2.destroyAllWindows()



    svmtrnr(usnm,mode)

# COUNTER = 0
# TOTAL = 0


# vs1 = VideoStream(src=0).start()
# # vs1 = VideoStream(usePiCamera=True).start()
# fileStream = False
# time.sleep(1.0)
# clsd = 0
# frameno = 0
# blink_data = []
# start = time.time()
# # loop over frames from the video stream
# while True:
#     # if this is a file video stream, then we need to check if
#     # there any more frames left in the buffer to process

#     if fileStream and not vs1.more():
#         break
    
 
#     # grab the frame from the threaded video file stream, resize
#     # it, and convert it to grayscale
#     # channels)
#     frame = vs1.read()
#     frame = imutils.resize(frame, width=450)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
#     # detect faces in the grayscale frame
#     rects = detector(gray, 0)

#     # loop over the face detections
#     for rect in rects:
#         # determine the facial landmarks for the face region, then
#         # convert the facial landmark (x, y)-coordinates to a NumPy
#         # array
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
 
#         # extract the left and right eye coordinates, then use the
#         # coordinates to compute the eye aspect ratio for both eyes
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = e_a_r(leftEye)
#         rightEAR =e_a_r(rightEye)
#         # average the eye aspect ratio together for both eyes
#         ear = (leftEAR + rightEAR) / 2.0
#         #blink_data.append([frameno, ear])
#         #frameno +=1

#         # compute the convex hull for the left and right eye, then
#         # visualize each of the eyes
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

#         # check to see if the eye aspect ratio is below the blink
#         # threshold, and if so, increment the blink frame counter
#         if ear < EYE_AR_THRESH:
#             COUNTER += 1
#             blink_data.append([frameno, ear, leftEAR, rightEAR, 0])
#             frameno += 1
            
 
#         # otherwise, the eye aspect ratio is not below the blink
#         # threshold
#         else:
#             # if the eyes were closed for a sufficient number of
#             # then increment the total number of blinks
            
#             blink_data.append([frameno, ear, leftEAR, rightEAR, 0])
#             if TOTAL == 0:
#                 if COUNTER >= EYE_AR_CONSEC_FRAMES_MIN and COUNTER <= EYE_AR_CONSEC_FRAMES_MAX:
#                     TOTAL = 1
                    
#                     for i in range(frameno-COUNTER, frameno,1):
#                         blink_data[i][-1] = 1
#                     print("\n bLINK AT :", (frameno - COUNTER), " for:",COUNTER ," frames\n" )
#                     IBT = 0


#             elif TOTAL > 0 :
                
#                 if COUNTER >= EYE_AR_CONSEC_FRAMES_MIN and COUNTER <= EYE_AR_CONSEC_FRAMES_MAX and IBT >= 2:
#                     TOTAL += 1
                    
#                     for i in range(frameno-COUNTER, frameno,1):
#                         blink_data[i][-1] = 1
#                     print("\n bLINK AT :", (frameno - COUNTER), " for:",COUNTER ," frames\n" )
#                 elif COUNTER <  EYE_AR_CONSEC_FRAMES_MIN and COUNTER > 0 :
#                     clsd = COUNTER
#                     frm = (frameno) - COUNTER
#                     blink_data[frameno-COUNTER][-1] = 0 
#                     print( " Not blink: {} , from frame: {}".format(clsd, frm) )
#                 elif COUNTER == 0:
#                     IBT += 1
#                 else:
#                     clsd = COUNTER
#                     frm = (frameno ) - COUNTER
#                     for i in range( (frameno - COUNTER), frameno, 1):
#                         blink_data[(frameno)-i][-1] = 0
#                     print( " Not blink: {} , from frame: {}".format(clsd, frm) )

#             frameno += 1
#             # reset the eye frame counter
#             COUNTER = 0
#         # draw the total number of blinks on the frame along with
#         # the computed eye aspect ratio for the frame
#         cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         cv2.putText(frame, " Blink type 2 Press 'q' after blinking  ", (1, 310),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(frame, "  sufficient number of times (at least 15) ", (1, 330),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    
#     # show the frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
 
#     # if key == ord("a"):
#     #     EYE_AR_CONSEC_FRAMES_MAX = 24                                                                 
#     #     EYE_AR_CONSEC_FRAMES_MIN = 2
#     #     print("\nMode A\n")
    
#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         pickle.dump(blink_data, open ("clean_s.pickle", "wb"))
#         end = time.time()
#        # print("\n Total frames captured :", frameno-1," in:",end - start ," seconds\n")
#         break
 
# # do a bit of cleanup
# cv2.destroyAllWindows()
# vs1.stop()
# svmtrnr('clean_s.pickle')
