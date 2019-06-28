# ASL-Complete-Alphabet-Recognition

To run with pretrained model data, download the files in the link and place them in the same directory as the python or jupyter file: 

https://drive.google.com/drive/folders/1rOtbhmVBv6_wRUTcFAjDEoRkN9Muk7VM?usp=sharing

We obtained the data from the following kaggle site:

https://www.kaggle.com/grassknoted/asl-alphabet

## Using the code

**The ASL recognition class**

To use the ASL recognizer, start by making an instance of the class. If you have the pretrained model downloaded, all you have to run is

```
asl = ASLRecognition()
```

If you would like to start with an untrained model instead, run

```
asl = ASLRecognition(load = False)
```

This will return a ResNet50 model with randomly initialized weights. 


**Training the Model**

If you would like to train the model on a new dataset, make sure you **do not** load the pretrained weights, then you can run 

```
asl.fitCNN("directory_of_new_dataset")
```

Make sure the data is organized by letter, as in all images for one sign should be in the same file inside the parent data file.

**Predicting** 

