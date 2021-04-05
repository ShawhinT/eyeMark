# HELPER FUNCTIONS FOR BLINK DETECT MAIN FUNCTION

# CODE AUTHORED BY: SHAWHIN TALEBI
# THE UNIVERSITY OF TEXAS AT DALLAS
# MULTI-SCALE INTEGRATED REMOTE SENSING AND SIMULATION (MINTS)

# Parts of code taken from reference: https://github.com/Danotsonof/facial-landmark-detection/blob/master/facial-landmark.ipynb
# Reference code authored by: Daniel Otulagun

# import modules
import cv2
# used for accessing url to download files
import urllib.request as urlreq
# used to access local directory
import os
# import modules for computing means and distances
import numpy as np
from numpy import linalg as LA

def makeFace(eyeStreamImage):
    # read smile image from file
    smile = cv2.imread('backend/smile.jpeg', 0)

    # set dimension for cropping right eye
    x, y, width, depth = 0, 720, 240, 240
    image_rightEye = eyeStreamImage[y:(y+depth), x:(x+width)]

    # set dimension for cropping left eye
    x, y, width, depth = 0, 240, 240, 240
    image_leftEye = eyeStreamImage[y:(y+depth), x:(x+width)]

    # create black background to surround face
    background_TB = np.zeros([100, 680], np.uint8)
    background_LR = np.zeros([600, 100], np.uint8)

    # concatenate eyes, smile, and background
    image_eyes = np.concatenate((image_rightEye, image_leftEye), axis=1)
    image_face = np.concatenate((image_eyes, smile), axis=0)

    image_face = np.concatenate((background_LR, image_face, background_LR), axis=1)
    image_face = np.concatenate((background_TB, image_face, background_TB), axis=0)

    return image_face

def getFaceDetector():
    # save face detection algorithm's url in haarcascade_url variable
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

    # save face detection algorithm's name as haarcascade
    haarcascade = "haarcascade_frontalface_alt2.xml"

    # chech if file is in working directory
    if (haarcascade in os.listdir(os.curdir)):
        # print("Face detector: File exists")
        pass
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(haarcascade_url, haarcascade)
        print("Face detector: File downloaded")

    return haarcascade

def findFaces(image_face):
    # download face detection algorithm
    haarcascade = getFaceDetector()

    # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(haarcascade)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(image_face)

    return faces

def getLandmarkDectector():
    # save facial landmark detection model's url in LBFmodel_url variable
    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

    # save facial landmark detection model's name as LBFmodel
    LBFmodel = "LFBmodel.yaml"

    # check if file is in working directory
    if (LBFmodel in os.listdir(os.curdir)):
        # print("Landmark detector: File exists")
        pass
    else:
        # download picture from url and save locally as lbfmodel.yaml
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
        print("Landmark detector: File downloaded")

    return LBFmodel

def getLandmarks(image_face):
    # find faces
    faces = findFaces(image_face)

    # download landmark detector
    LBFmodel = getLandmarkDectector()

    # create an instance of the Facial landmark Detector with the model
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)

    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(image_face, faces)

    return landmarks

def getEyeLandmarks(landmarks):
    # define indicies corresponding to eye landmarks of interest
    rightEyeHindx = [36, 39]
    rightEyeVindx = [37, 38, 40, 41]
    leftEyeHindx = [42, 45]
    leftEyeVindx = [43, 44, 46, 47]

    # initialize arrays to store right and left eye landmark locations
    # store 2D eye landmarks in a 12 by 2 array in the following order:
    # ---> 1. horizontal right eye landmarks (ordered left to right)
    # ---> 2. vertical right eye landmarks
    # ---> 3. horizontal left eye landmarks (ordered left to right)
    # ---> 4. vertical left eye landmarks
    eyeLandmarks = np.zeros([12,2])

    # store right and left eye landmark locations in arrays
    for landmark in landmarks:
        # right eye horizontal landmarks
        j=0
        for i in rightEyeHindx:
            eyeLandmarks[j] = landmark[0][i]
            j=j+1
        # right eye vertical landmarks
        for i in rightEyeVindx:
            eyeLandmarks[j] = landmark[0][i]
            j=j+1
        # left eye horizontal landmarks
        for i in leftEyeHindx:
            eyeLandmarks[j] = landmark[0][i]
            j=j+1
        # left eye vertical landmarks
        for i in leftEyeVindx:
            eyeLandmarks[j] = landmark[0][i]
            j=j+1

    return eyeLandmarks


def computeWidth(eyeHLocs):
    width = LA.norm(eyeHLocs[0,:]-eyeHLocs[1,:])

    return width

def computeHeight(eyeVLocs):
    bottomCenter = np.mean(eyeVLocs[0:2,:])
    topCenter = np.mean(eyeVLocs[2:4,:])

    height = LA.norm(topCenter-bottomCenter)

    return height

def computeScaledSeparation(eyeHLocs, eyeVLocs):

    width = computeWidth(eyeHLocs)
    height = computeHeight(eyeVLocs)

    scaledSeparation = height/width

    return scaledSeparation, height, width

def getSeparations(eyeLandmarks):

    # intialize array to store eye separations
    # store eye separations in the following format:
    # -----------------|-- rightEye --|-- leftEye --|
    # scaledSeparation |       x      |      x      |
    #     height       |       x      |      x      |
    #     width        |       x      |      x      |
    # ----------------------------------------------|
    separations = np.zeros([3,2])

    # get separations for right eye
    separations[0,0], separations[1,0], separations[2,0] = \
        computeScaledSeparation(eyeLandmarks[0:2,:], eyeLandmarks[2:6,:])

    # get separations for left eye
    separations[0,1], separations[1,1], separations[2,1] = \
        computeScaledSeparation(eyeLandmarks[6:8,:], eyeLandmarks[8:12,:])

    return separations

# def computeEyeSeperation(eyeLocs):
#
#     # compute average location
#     meanEyeLoc = np.mean(eyeLocs,  axis=0)
#
#     # create array to store euclidean distances of each eye landmark from mean
#     temp = []
#     # store distance of each eye landmark from mean in temp
#     for i in range(len(eyeLocs)):
#         temp.append(LA.norm(eyeLocs[i,:]-meanEyeLoc))
#
#     # compute mean distance from center
#     eyeSeparation = np.mean(temp)
#
#     return eyeSeparation

# def computeMeanSeparation(rightEyeSeparation, leftEyeSeparation):
#
#     # compute mean separation
#     meanSeparation = np.mean([leftEyeSeparation,  rightEyeSeparation])
#
#     return meanSeparation

# def eyeSeparation(landmarks):
#     # get eye landmark locations
#     rightEyeLocs, leftEyeLocs = getEyeLandmarks(landmarks)
#
#     # compute eye separation
#     rightEyeSeparation = computeEyeSeperation(rightEyeLocs)
#     leftEyeSeparation = computeEyeSeperation(leftEyeLocs)
#
#     # compute mean separation
#     meanSeparation = computeMeanSeparation(rightEyeSeparation, leftEyeSeparation)
#
#     return meanSeparation
