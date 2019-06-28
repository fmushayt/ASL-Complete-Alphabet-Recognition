#!/usr/bin/env python
# coding: utf-8

# # ASL Complete Alphabet Recognition
# 

# In[2]:


# Faisal Mushayt
# Myles Cork
# Deemah Al Dulaijan
# ---------------------------------------------------------------
get_ipython().run_line_magic('matplotlib', 'inline')
import keras
import cv2
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists, expanduser
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model
from collections import deque


# ## Helper functions

# In[3]:


def myFrameDifferencing(prev, curr):
    '''
       Difference between last two frames 
    '''
    dst = cv2.absdiff(prev, curr)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, dst = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
    return dst

# --------------------------------------------------------------

def myMotionEnergy(mh):
    '''
        Captures motion Energy by setting any pixel observed in the 
        last 3 frames to white 
    '''
    mh0 = mh[0]
    mh1 = mh[1]
    mh2 = mh[2]
    dst = np.zeros((mh0.shape[0], mh0.shape[1], 1), dtype = "uint8")
    for i in range(mh0.shape[0]):
        for j in range(mh0.shape[1]):
            if mh0[i,j] == 255 or mh1[i,j] == 255 or mh2[i,j] == 255:
                dst[i,j] = 255
    return dst

# --------------------------------------------------------------

def getFrame():
    '''
        Records video and takes either static or dynamic sign.
        For static, returns image of sign
        For dynamic, returns motion energy image  
    '''
    
    # Initiate video capture
    cap = cv2.VideoCapture(0)
    
    
    # While no command has been issued, take video
    while True:
        
        # Get current frame
        _, curr_frame = cap.read()
        
        # Check for command key
        k = cv2.waitKey(1)
        
        # If esc hit, terminate prediction
        if k%256 == 27:
            print("Escape hit, closing...")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
            
        # If 's' hit, take static sign
        elif k%256 == ord('s'):
            motion = False
            img = cv2.resize(curr_frame, (224,224))
            break
        
        # If 'd' hit, take dynamic sign (j or z)
        elif k%256 == ord('d'):
            motion = True
            
            # Record current frame
            _, prev_frame = cap.read()
                    
            # Motion history set up
            fMH1 = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 1), dtype = "uint8")
            fMH2 = fMH1.copy()
            fMH3 = fMH1.copy()
            myMotionHistory = deque([fMH1, fMH2, fMH3])             
            while True:
                # Get current frame
                _, curr_frame = cap.read()
                
                # Keep track of frame differences
                frameDest = myFrameDifferencing(prev_frame, curr_frame)                
                myMotionHistory.popleft()
                myMotionHistory.append(frameDest)

                k = cv2.waitKey(1)
                # If 'd' hit again, dynamic sign is done
                if k%256 == ord('d'):
                    break
                
                # Display frame
                cv2.imshow("Detect Sign", curr_frame)
            
            # Create motion energy image
            img = myMotionEnergy(myMotionHistory)
            break
        
        # If no command issued, do nothing
        else: pass
        
        # Display frame
        cv2.imshow("Detect Sign", curr_frame)
    
    # Release cap and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Return whether motion was indicated and the appropriate image
    return motion, img

# --------------------------------------------------------------

def predictJZ(myMH):
    '''
        Differentiates between the motion energy of J and Z
        by pattern matching the motion history to premade 
        templates
    '''
    # read in "J" templates and intialize variables
    j_temp1 = cv2.imread("templates/j_1.jpg", 0)
    j_temp2 = cv2.imread("templates/j_2.jpg", 0)
    j_temp3 = cv2.imread("templates/j_3.jpg", 0)
    j_temp4 = cv2.imread("templates/j_4.jpg", 0)
    j_temp5 = cv2.imread("templates/j_5.jpg", 0)
    j_thresh = 0.35

    # read in "Z" templates and initialize variables
    z_temp1 = cv2.imread("templates/z_1.jpg", 0)
    z_temp2 = cv2.imread("templates/z_2.jpg", 0)
    z_temp3 = cv2.imread("templates/z_3.jpg", 0)
    z_temp4 = cv2.imread("templates/z_4.jpg", 0)
    z_temp5 = cv2.imread("templates/z_5.jpg", 0)
    z_thresh = 0.95

    action = "Nothing"

    # detect j
    # Comparison results of 5 different j templates
    jres1 = cv2.matchTemplate(myMH, j_temp1, cv2.TM_CCOEFF_NORMED)
    jres2 = cv2.matchTemplate(myMH, j_temp2, cv2.TM_CCOEFF_NORMED)
    jres3 = cv2.matchTemplate(myMH, j_temp3, cv2.TM_CCOEFF_NORMED)
    jres4 = cv2.matchTemplate(myMH, j_temp4, cv2.TM_CCOEFF_NORMED)
    jres5 = cv2.matchTemplate(myMH, j_temp5, cv2.TM_CCOEFF_NORMED)

    # find locations of points that are similar to the templates using a threshold
    jloc1 = np.where( jres1 >= j_thresh)
    jloc2 = np.where( jres2 >= j_thresh)
    jloc3 = np.where( jres3 >= j_thresh)
    jloc4 = np.where( jres4 >= j_thresh)
    jloc5 = np.where( jres5 >= j_thresh)

    # detect z
    # Comparison results of 5 different j templates
    zres1 = cv2.matchTemplate(myMH, z_temp1, cv2.TM_CCOEFF_NORMED)
    zres2 = cv2.matchTemplate(myMH, z_temp2, cv2.TM_CCOEFF_NORMED)
    zres3 = cv2.matchTemplate(myMH, z_temp3, cv2.TM_CCOEFF_NORMED)
    zres4 = cv2.matchTemplate(myMH, z_temp4, cv2.TM_CCOEFF_NORMED)
    zres5 = cv2.matchTemplate(myMH, z_temp5, cv2.TM_CCOEFF_NORMED)

    # find locations of points that are similar to the templates using a threshold
    zloc1 = np.where( zres1 >= z_thresh)
    zloc2 = np.where( zres2 >= z_thresh)
    zloc3 = np.where( zres3 >= z_thresh)
    zloc4 = np.where( zres4 >= z_thresh)
    zloc5 = np.where( zres5 >= z_thresh)

    # if there are similar points, classify the sign as J
    flagj = []
    for pt in zip(*jloc1[::-1]):
        if pt is not None:
            flagj.append(True)
    for pt in zip(*jloc2[::-1]):
        if pt is not None:
            flagj.append(True)
    for pt in zip(*jloc3[::-1]):
        if pt is not None:
            flagj.append(True)
    for pt in zip(*jloc4[::-1]):
        if pt is not None:
            flagj.append(True)
    for pt in zip(*jloc5[::-1]):
        if pt is not None:
            flagj.append(True)
    
    # if there are similar points, classify the sign as Z
    flagz = []
    for pt in zip(*zloc1[::-1]):
        if pt is not None:
            flagz.append(True)
    for pt in zip(*zloc2[::-1]):
        if pt is not None:
            flagz.append(True)
    for pt in zip(*zloc3[::-1]):
        if pt is not None:
            flagz.append(True)
    for pt in zip(*zloc4[::-1]):
        if pt is not None:
            flagz.append(True)
    for pt in zip(*zloc5[::-1]):
        if pt is not None:
            flagz.append(True)
    if len(flagj) > len(flagz):
        action = "J"
    else:
        action = "Z"

    return action


# ## ASL Recognizer Class

# In[4]:


class ASLRecognition():
    
    def __init__(self, load = True):
        ''' 
            Initializes ResNet model either randomly or load existing
            
            Params:
                Loss = categorical crossentropy (log loss)
                Optimizer = adam
        '''
        
        if load == True:
            
            # Load model and history data
            self.model = load_model('asl_model.h5')
            with open('asl_history.json', 'r') as fp:
                self.history = json.load(fp)
            print("Model loaded")
            
        else:
            # Initialize ResNet50 model
            self.history = dict()
            self.model = ResNet50(weights = None, classes = 24)
            
            '''
            The best result we found was without applying regularization
            manually. We go into more detail on this in our report.
            
            
            # Apply weight decay terms to l2 regularize
            # Taken from https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
            decay = 0.001
            for layer in self.model.layers:
                if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                    layer.add_loss(keras.regularizers.l2(decay)(layer.kernel))
                if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                    layer.add_loss(keras.regularizers.l2(decay)(layer.bias))
            '''
            
            # Configure the network with the Adam optimizer and logloss 
            self.model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics=['accuracy'])
            print("Model initialized")
            
            
        return
        
    
    
    
    def fitCNN(self, directory):
        ''' input: filepath (directory) name
            output: None
            
            Fits ResNet50 model onto provided data and evaluate metrics
            
            Parameters:
                Validation percentage = 20%
                Epochs = 50       
        '''
        
        # Randomly split images into training and validation subsets using a 80:20 ratio
        vsplit = 0.2
        img_gen = image.ImageDataGenerator(validation_split = vsplit)
        
        # Generate training and validation data after preprocessing images
        train_batch = img_gen.flow_from_directory(directory, target_size = (224, 224), subset = "training")
        valid_batch = img_gen.flow_from_directory(directory, target_size = (224, 224), subset = "validation")
        
        # train the model on the data and record history
        self.history = self.model.fit_generator(train_batch, epochs=50, steps_per_epoch = 1788, validation_data= valid_batch, validation_steps = 447)
        self.history = self.history.history
        return
     
        
        
    
    def predict(self, img_path=None):
        ''' input: Image path
            output: None
            
             Print the predicted labels with the highest probability.
             If no image path passed in, starts video frame that waits
             for command:
                 Esc: cancel prediction
                 s: take static image for (a-y) prediction not including J
                 d: take recording until d hit again, captures dynamic sign
                    and predicts either J or Z
        '''
        
        # Decide whether to predict from camera or existing image
        if not img_path:
            # Capture frame and decide how to proceed
            motion, img = getFrame()
        else:
            # Process existing image
            motion = False
            img = image.load_img(img_path, target_size=(224, 224))
        
        
        
        # If we observe motion, classify as j or z
        if motion:
            sign = predictJZ(img)
            print('Predicted: ' + sign)
            return
        
        # Otherwise, use ResNet50 to classify a-y (!= j)
        else:
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            preds = self.model.predict(img)
            print('Predicted:', self.decode_predictions(preds))        
            
        return
    
    
    
    
    def decode_predictions(self, preds, top=3):
        ''' input:
                  preds -> array of prediction results from keras predict
            output:
                  decoded -> dictionary of 'top' labels and corresponding probailities
        '''
        
        # Set labels
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
       
        # Add the labels and their probabilities to a dictionary
        decoded = dict()
        for i in range(24): decoded[labels[i]] = preds[0][i]

        # Obtain the "top" predictions and return them
        preds[0].sort()
        TOP = preds[0][-top:]
        decoded = dict((k,v) for k,v in decoded.items() if v in TOP)
        return decoded
       
    
    def save(self):
        ''' 
            Saves new model in H5 file and history in json file
        '''
        
        self.model.save('asl_model_new.h5')
        with open('asl_history_new.json', 'w') as fp:
            json.dump(self.history, fp)
            
        print("Model saved")
        return

