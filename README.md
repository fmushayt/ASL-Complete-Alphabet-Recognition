# ASL-Complete-Alphabet-Recognition

To run with pretrained model data, download the files in the link and place them in the same directory as the python or jupyter file: 

https://drive.google.com/drive/folders/1rOtbhmVBv6_wRUTcFAjDEoRkN9Muk7VM?usp=sharing

We obtained the data from the following kaggle site:

https://www.kaggle.com/grassknoted/asl-alphabet

## Abstract

*Insert Abstract Here*

## Using the code

_**The ASL recognition class**_

To use the ASL recognizer, start by making an instance of the class. If you have the pretrained model downloaded, all you have to run is

```
model = ASLRecognition()
```

If you would like to start with an untrained model instead, run

```
model = ASLRecognition(load = False)
```

This will return a ResNet50 model with randomly initialized weights. 




_**Training the Model**_

If you would like to train the model on a new dataset, make sure you **do not** load the pretrained weights, then you can run 

```
model.fitCNN("directory_of_new_dataset")
```

Make sure the data is organized by letter, as in all images for one sign should be in the same file inside the parent data file.



_**Predicting**_

To predict on a new sign, you can either predict on an existing image of a static sign,

```
model.predict(img_path = "path_of_img")
```

or use your webcam to either take a static or dynamic sign,

```
model.predict()
```

This starts a video frame that waits for a command:
                
       Esc: cancel prediction
       s: take static image for (a-y) prediction not including J
       d: take recording until d hit again, captures dynamic sign and predicts either J or Z



_**Saving a new model**_

All you have to do to save a newly trained model is

```
model.save()
```

This will save two new files, one with the model data (weights, etc.) called **asl_model_new.h5** and another with the metric history data called **asl_history_new.json**
