# MAIN FUNCTION: EYE MARK

# FUNCTION TO GET EYE LANDMARKS, WIDTH AND HEIGHT OF EYE LID SEPARATION, AND EYE
# LID ASPECT RATIO FROM EYESTREAM VIDEO TAKEN WITH TOBII PRO GLASSES 2

# CODE AUTHORED BY: SHAWHIN TALEBI
# THE UNIVERSITY OF TEXAS AT DALLAS
# MULTI-SCALE INTEGRATED REMOTE SENSING AND SIMULATION (MINTS)

# Parts of code taken from reference: https://github.com/Danotsonof/facial-landmark-detection/blob/master/facial-landmark.ipynb
# Reference code authored by: Daniel Otulagun

# import function dependencies 
from functions import *

# function to
def eyeMark(image_gray):

    # make artifical face from eye stream image
    image_face = makeFace(image_gray)

    # get landmarks
    landmarks = getLandmarks(image_face)

    # get locations of eye landmarks
    eyeLandmarks = getEyeLandmarks(landmarks)

    # get eye scaled separations for both eyes, along with heights and widths
    separations = getSeparations(eyeLandmarks)

    return eyeLandmarks, separations
