from keras.applications import MobileNet
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.models import Model
import cv2
import numpy as np
#Load the MobileNet model
img_rows = 224
img_cols = 224  
model = MobileNet(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

#for printing  layers of the model
for (i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)

#shows the trainable layers

model.layers[0].trainable


model.output
# to Freez the layers(important line)

for layer in model.layers:
    layer.trainable = False
#Storing the output layer in a variable

model_output = model.output

model_output = Flatten()(model_output)

#for adding extra layers

model_output = Dense(units = 512 , activation='relu' )(model_output)
model_output = Dense(units= 256, activation='relu' )(model_output)
model_output = Dense(units= 1 , activation='sigmoid' )(model_output)


#shows the input and output layers

model= Model(inputs=model.input , outputs = model_output)

#compile the model

model.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

#for training the model after augementation

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'C:/Users/Dell/Desktop/mloops/Images/train/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'C:/Users/Dell/Desktop/mloops/Images/validation/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=70,
        epochs=1,
        validation_data=test_set,
        validation_steps=10)




# for Saving and loading the model

model.save('face_recog_mobilenet.h5')
from keras.models import load_model
classifier = load_model('face_recog_mobilenet.h5')




#to Predict the Random Images using face recognization model

import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

human_dict = {"[0]": "rani" 
              }

human_dict_n = {"rani": "rani"
                }

def draw_test(name, pred, im):
    human = human_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, human, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + human_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage("C:/Users/Dell/Desktop/mloops/Images/validation/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # to Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()

