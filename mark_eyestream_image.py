# SAMPLE CODE USING EYE MARK FUNCTION ON EYE STREAM VIDEO FROM TOBII PRO GLASSES 2

# CODE AUTHORED BY: SHAWHIN TALEBI
# THE UNIVERSITY OF TEXAS AT DALLAS
# MULTI-SCALE INTEGRATED REMOTE SENSING AND SIMULATION (MINTS)

# import modules and function dependencies
import cv2
from functions import *
from eyeMark import eyeMark

# read image with openCV
image = cv2.imread('sampleImages/closed/2020_06_04_T01_U00T_Tobii01_eyesstream_frame_425.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

########
eyeLandmarks, separations = eyeMark(image_gray)
print('\n')
print(eyeLandmarks)
print('\n')
print(separations)
