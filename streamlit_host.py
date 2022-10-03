# import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import imutils
from displayTumor import *

model = tf.keras.models.load_model("./curr_mode.hdf5")


st.title("Brain Tumor Detection")
st.write("By Abhay | Akash | Aman | Himanshu")

st.write("NSUT BTP Project-2022")
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

class_dict = {0: "glioma_tumor", 1: "meningioma_tumor", 2: "glioma_tumor", 3: "pituitary_tumor"}


if uploaded_file is not None:

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))

    # From here onwards there is another model to predict
    model1 = tf.keras.models.load_model('set2detection.h5')
    image=resized
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv.threshold(gray, 45, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    image = cv.resize(new_image, dsize=(240, 240), interpolation=cv.INTER_CUBIC)
    image = image / 255.

    image = image.reshape((1, 240, 240, 3))

    res = model1.predict(image)
    flag=0
    if(res>0.5):
        flag=1
    else:
        flag=0
        
    # till here

    # Now do something with the image! For example, let's display it:    
    firstdisplay = opencv_image
    # firstdisplay.reshape((1, 240, 240, 3))
    st.image(new_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()

        curr_cat=class_dict [prediction]
        if(flag==0):
            st.title("No Tumor is Present")
        # elif(prediction==2):
        #     st.title("No Tumor is Present")
        else:
            st.title("Tumor is Present and category is {}".format(curr_cat))

            # if(prediction != 2):    
            img=uploaded_file
            Img = opencv_image
            # curImg = np.array(img)
            curImg=opencv_image

            gray = cv2.cvtColor(np.array(Img), cv2.COLOR_BGR2GRAY)
            [ret, thresh] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            curImg = opening

            # sure background area
            sure_bg = cv2.dilate(curImg, kernel, iterations=3)

            # Finding sure foreground area
            # curImg = image.img_to_array(curImg, dtype='uint8')
            dist_transform = cv2.distanceTransform(curImg, cv.DIST_L2, 5)
            [ret, sure_fg] = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Find unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg,dtype=cv2.CV_32F)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now mark the region of unknown with zero
            markers[unknown == 255] = 0   
            markers = cv2.watershed(opencv_image, markers)

            opencv_image[markers == -1] = [255, 0, 0]

            tumorImage = cv2.cvtColor(opencv_image, cv2.COLOR_HSV2BGR)
            curImg = tumorImage            
            st.image(curImg)
