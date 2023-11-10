# Artificial-Intelligence-Challenge

**Challenge 1: Build a model with different classes from QuickDraw.io by Google and deploy the model onto a Flask website.**
#### <a name="_eqtgir9o4ekz"></a>**1.             Evaluate the data**
The ﬁrst thing we always need in a machine learning model is data. Data preparation can take up to 80% of the time spent on an ML project. I used is in the Quick, Draw! Dataset by google creative lab.The Quick Draw Dataset is a collection of 50 million drawings across[ ](https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt)[345 categories](https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt), contributed by players of the game[ ](https://quickdraw.withgoogle.com)[Quick, Draw!](https://quickdraw.withgoogle.com). The drawings were captured as timestamped vectors, tagged with metadata including what the player was asked to draw and in which country the player was located. You can browse the recognized drawings on[ ](https://quickdraw.withgoogle.com/data)[quickdraw.withgoogle.com/data](https://quickdraw.withgoogle.com/data).The raw data is available as [ndjson](http://ndjson.org/) files separated by category, in the following format:


|**Key**|**Type**|**Description**|
| :-: | :-: | :-: |
|key\_id|64-bit unsigned integer|A unique identifier across all drawings.|
|word|string|Category the player was prompted to draw.|
|recognized|boolean|Whether the word was recognized by the game.|
|timestamp|datetime|When the drawing was created.|
|countrycode|string|A two letter country code ([ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)) of where the player was located.|
|drawing|string|A JSON array representing the vector drawing|

Each line contains one drawing. Here's an example of a single drawing:

|`  `{ <br>`    `"key\_id":"5891796615823360",<br>`    `"word":"nose",<br>`    `"countrycode":"AE",<br>`    `"timestamp":"2017-03-01 20:41:36.70725 UTC",<br>`    `"recognized":true,<br>`    `"drawing":[[[129,128,129,129,130,130,131,132,132,133,133,133,133,...]]]<br>`  `}|
| :- |
#### <a name="_vhaopys92tp3"></a>**2. Creating a class\_names.txt** 
In this stage, I randomly selected 3 classes out of 345 categories available in the dataset I mentioned earlier. So I created a file format called text, in which there are 3 categories I chose including car, table and laptop. This is how my class\_names.txt look on GoogleColab: 

Then I imported this text file in my GoogleColab : 



It will locate those layers and download all the photographs in the desired portion when it reads the layers that I have recorded in the text file. The photographs will all be kept in a folder called “ **data** ” that I made, as shown in the image above in the third line. The files within that folder are in the form of root files, which include all of the images in numpy format.

#### <a name="_hr0z6orn2y0v"></a>**3. Approach the images**


|def load\_data(root, vfold\_ratio=0.2, max\_items\_per\_class= 4000 ):<br>`    `all\_files = glob.glob(os.path.join(root, '\*.npy'))<br><br>`    `*#initialize variables*<br>`    `x = np.empty([0, 784])<br>`    `y = np.empty([0])<br>`    `class\_names = []<br><br>`    `*#load each data file*<br>`    `for idx, file in enumerate(all\_files):<br>`        `data = np.load(file)<br>`        `data = data[0: max\_items\_per\_class, :]<br>`        `labels = np.full(data.shape[0], idx)<br><br>`        `x = np.concatenate((x, data), axis=0)<br>`        `y = np.append(y, labels)<br><br>`        `class\_name, ext = os.path.splitext(os.path.basename(file))<br>`        `class\_names.append(class\_name)<br><br>`    `data = None<br>`    `labels = None<br><br>`    `*#randomize the dataset*<br>`    `permutation = np.random.permutation(y.shape[0])<br>`    `x = x[permutation, :]<br>`    `y = y[permutation]<br><br>`    `*#separate into training and testing*<br>`    `vfold\_size = int(x.shape[0]/100\*(vfold\_ratio\*100))<br><br>`    `x\_test = x[0:vfold\_size, :]<br>`    `y\_test = y[0:vfold\_size]<br><br>`    `x\_train = x[vfold\_size:x.shape[0], :]<br>`    `y\_train = y[vfold\_size:y.shape[0]]<br>`    `return x\_train, y\_train, x\_test, y\_test, class\_names|
| :- |

This code defines a function load\_data that loads and preprocesses a dataset stored in multiple NumPy files (.npy files) from a specified directory (root). The dataset appears to consist of image data where each image is represented as a 1D array of size 784 (28x28 pixels flattened).

Here's a step-by-step explanation of the code:

1. Import necessary libraries and modules:


|import os<br>import glob<br>import numpy as np|
| :- |

1. Define the “ **load\_data** “  function with parameters “ **root** “, **“ vfold\_ratio ”, and “ max\_items\_per\_class”** :


|def load\_data(root, vfold\_ratio=0.2, max\_items\_per\_class=4000):|
| :- |

1. Use **“ glob “**  to get a list of all files with the **“ .npy ”** extension in the specified directory **(“ root ”)**:


|all\_files = glob.glob(os.path.join(root, '\*.npy'))|
| :- |

1. Initialize variables for data **(“ x “)**, labels **(“ y “)**, and class names **(“ class\_names ”)**:


|x = np.empty([0, 784])<br>y = np.empty([0])<br>class\_names = []|
| :- |

1. Loop through each file in the list of .npy files:

|for idx, file in enumerate(all\_files):|
| :- |

1. Load the data from the current file and select a subset **(“ max\_items\_per\_class “)** of rows:

|data = np.load(file)<br>data = data[0: max\_items\_per\_class, :]|
| :- |

1. Create an array of labels **(“ labels “)** for the current data:

|labels = np.full(data.shape[0], idx)|
| :- |

1. Concatenate the current data and labels to the overall data **(“x“)** and labels **(“ y “)** arrays:


|x = np.concatenate((x, data), axis=0)<br>y = np.append(y, labels)|
| :- |

1. Extract the class name from the file name and add it to the list of class names **(“class\_names “)**:


|class\_name, ext = os.path.splitext(os.path.basename(file))<br>class\_names.append(class\_name)|
| :- |

1. Set temporary variables **(“ data and labels “)** to None to free up memory:


|data = None<br>labels = None|
| :- |

1. Randomize the dataset by generating a random permutation of indices and rearranging the data accordingly:


|permutation = np.random.permutation(y.shape[0])<br>x = x[permutation, :]<br>y = y[permutation]|
| :- |

1. Calculate the size of the validation fold **( “ vfold\_size “)** based on the specified ratio 


|(vfold\_ratio) and split the data into training and testing sets:<br>vfold\_size = int(x.shape[0] / 100 \* (vfold\_ratio \* 100))<br><br>x\_test = x[0:vfold\_size, :]<br>y\_test = y[0:vfold\_size]<br><br>x\_train = x[vfold\_size:x.shape[0], :]<br>y\_train = y[vfold\_size:y.shape[0]]|
| :- |

1. Return the training and testing data along with the list of class names:


|return x\_train, y\_train, x\_test, y\_test, class\_names|
| :- |

In summary, the function loads image data from multiple files, preprocesses it by selecting a subset and randomizing the order, and then splits it into training and testing sets. The function also returns the list of class names corresponding to the loaded data.




Next, to prepare for input data, we will have to synchronize the images in the data set into one form. In the image below I have put my entire data set into 28x28 format (which means 28 horizontally and 28 vertically). Next I will normalize the values in the image to the range (0,1). For the reason that they will help our model learn and update good weights and converge faster. They will avoid situations where the model learns noisy values. For example, large values will be quite more sensitive than small values. The final step is to label each layer that we downloaded earlier. 


|x\_train, y\_train, x\_test, y\_test, class\_names = load\_data('data')<br>num\_classes = len(class\_names)<br>image\_size = 28|
| :- |


|*# Reshape and normalize*<br>x\_train = x\_train.reshape(x\_train.shape[0], image\_size, image\_size, 1).astype('float32')<br>x\_test = x\_test.reshape(x\_test.shape[0], image\_size, image\_size, 1).astype('float32')<br><br>x\_train /= 255.0<br>x\_test /= 255.0<br><br>*# Convert class vectors to class matrices*<br>y\_train = keras.utils.to\_categorical(y\_train, num\_classes)<br>y\_test = keras.utils.to\_categorical(y\_test, num\_classes)|
| :- |


#### <a name="_ax8dbw4amjqk"></a>**4. Define the model**

At this stage we will start creating a CNN model for image classification, compiling it with a suitable loss function Categorical CrossEntropy and optimizer Adam . Note that for this code to work, you need to have the appropriate libraries imported, such as TensorFlow and Keras. Additionally, the code assumes a 3-class classification task based on the last layer having 3 neurons with softmax activation.

\# Define model

model = keras.Sequential()

model.add(layers.Convolution2D(16, (3, 3),

`                        `*padding*='same',

`                        `*input\_shape*=x\_train.shape[1:], *activation*='relu'))

model.add(layers.MaxPooling2D(*pool\_size*=(2, 2)))

model.add(layers.Convolution2D(32, (3, 3), *padding*='same', *activation*= 'relu'))

model.add(layers.MaxPooling2D(*pool\_size*=(2, 2)))

model.add(layers.Convolution2D(64, (3, 3), *padding*='same', *activation*= 'relu'))

model.add(layers.MaxPooling2D(*pool\_size* =(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, *activation*='relu'))

model.add(layers.Dense(3, *activation*='softmax'))

\# Train model

adam = tf.optimizers.Adam()

model.compile(*loss*='categorical\_crossentropy',

`              `*optimizer*=adam,

`              `*metrics*=['accuracy'])

print(model.summary())


|Model: "sequential"<br>\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_<br>` `Layer (type)                Output Shape              Param *#*   <br>=================================================================<br>` `conv2d (Conv2D)             (None, 28, 28, 16)        160       <br>                                                                 <br>` `max\_pooling2d (MaxPooling2  (None, 14, 14, 16)        0         <br>` `D)                                                              <br>                                                                 <br>` `conv2d\_1 (Conv2D)           (None, 14, 14, 32)        4640      <br>                                                                 <br>` `max\_pooling2d\_1 (MaxPoolin  (None, 7, 7, 32)          0         <br>` `g2D)                                                            <br>                                                                 <br>` `conv2d\_2 (Conv2D)           (None, 7, 7, 64)          18496     <br>                                                                 <br>` `max\_pooling2d\_2 (MaxPoolin  (None, 3, 3, 64)          0         <br>` `g2D)                                                            <br>                                                                 <br>` `flatten (Flatten)           (None, 576)               0         <br>                                                                 <br>` `dense (Dense)               (None, 128)               73856     <br>                                                                 <br>` `dense\_1 (Dense)             (None, 3)                 387       <br>                                                                <br>...<br>Trainable params: 97539 (381.01 KB)<br>Non-trainable params: 0 (0.00 Byte)<br>\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_<br>None|
| :- |

#### <a name="_nla5lwkb4pxp"></a>**5. Training and testing** 

history= model.fit(*x* = x\_train, *y* = y\_train, *validation\_split*=0.2, *batch\_size* = 256, *verbose*=2, *epochs*=5)

|Epoch 1/5<br>30/30 - 7s - loss: 0.5414 - accuracy: 0.8454 - val\_loss: 0.2267 - val\_accuracy: 0.9120 - 7s/epoch - 224ms/step<br>Epoch 2/5<br>30/30 - 12s - loss: 0.1783 - accuracy: 0.9396 - val\_loss: 0.1543 - val\_accuracy: 0.9495 - 12s/epoch - 388ms/step<br>Epoch 3/5<br>30/30 - 6s - loss: 0.1259 - accuracy: 0.9602 - val\_loss: 0.1158 - val\_accuracy: 0.9604 - 6s/epoch - 203ms/step<br>Epoch 4/5<br>30/30 - 6s - loss: 0.1021 - accuracy: 0.9690 - val\_loss: 0.1099 - val\_accuracy: 0.9615 - 6s/epoch - 208ms/step<br>Epoch 5/5<br>30/30 - 5s - loss: 0.0892 - accuracy: 0.9710 - val\_loss: 0.0919 - val\_accuracy: 0.9661 - 5s/epoch - 169ms/step|
| :- |

#### <a name="_q5k722dhrxma"></a>**6. Overall assessment**

In order to assess the model's goodness following training, I utilized my training model to apply it to the test data set (this is a data set that the model has not yet seen). The outcomes were quite good. I will next create two graphs to assess the parameters of loss and accuracy because, in deep learning model training, the goal is always to reduce the loss function and maximize accuracy. The intended outcome was shown in the two graphs below. 



#### <a name="_fmr6apxh9weu"></a>**7. Testing the result**

#### <a name="_w5p1208bgkgv"></a>**8. Deploy the model**
The overall purpose of this code is to create a simple web application for image recognition using a pre-trained deep learning model. Users can upload an image, and the server will use the model to predict and return the class label for the content of the image. 

from flask import Flask, render\_template, request, jsonify

import base64

import tensorflow as tf

import cv2

import numpy as np



app = Flask(\_\_name\_\_)

model = tf.keras.models.load\_model('model4.h5')

model.make\_predict\_function()

with open('class\_names.txt') as file:

`    `class\_labels = file.read().splitlines()

@app.route('/')

def index():

`    `return render\_template('index.html')

@app.route('/recognize', *methods* = ['POST'])

def recognize():

`    `if request.method =='POST':

`        `print('Receive image and predict what it is')

`        `data = request.get\_json()

`        `imageBase64 = data['image']

`        `imgBytes = base64.b64decode(imageBase64)

`        `with open('temp.jpg', 'wb') as temp:

`            `temp.write(imgBytes)

`        `image = cv2.imread('temp.jpg')

`        `image = cv2.resize(image,(28,28),*interpolation* = cv2.INTER\_AREA)

`        `image\_gray = cv2.cvtColor(image,cv2.COLOR\_BGR2GRAY)

`        `image\_prediction = np.reshape(image\_gray,(28,28,1))

`        `image\_prediction = (255-image\_prediction.astype('float')) /255

`        `prediction = np.argmax(model.predict(np.array([image\_prediction])),*axis* = -1)

`        `#CHẠY PREDICTION

`        `return jsonify({

`            `'prediction':class\_labels[prediction[0]],

`            `'status': True

`        `})

if \_\_name\_\_ == '\_\_main\_\_':

`    `app.run(*debug* = True)


Below is the result of the entire process in this project with three classes: 


