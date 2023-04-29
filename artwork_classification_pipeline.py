# import packages
import os
import json
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import *
from tqdm import tqdm, tqdm_notebook

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from numpy.random import seed
seed(1)

# data preprocessing
print(os.listdir("images_genre"))
genre = pd.read_csv('genre.csv')

# Sort artists by number of paintings
genre = genre.sort_values(by=['paintings'], ascending=False)

# Create a dataframe with genre having more than 150 paintings
genre_top = genre[genre['paintings'] >= 150].reset_index()
genre_top = genre_top[['genre', 'paintings']]
#genre_top['class_weight'] = max(genre_top.paintings)/artists_top.paintings
genre_top['class_weight'] = genre_top.paintings.sum() / (genre_top.shape[0] * genre_top.paintings)
genre_top

print(sum(genre_top['paintings']))

# Set class weights - assign higher weights to underrepresented classes
class_weights = genre_top['class_weight'].to_dict()
class_weights

# Explore images of top genre
images_dir = 'images_genre'
genre_dirs = os.listdir(images_dir)
top_genre = genre_top['genre']

# See if all directories exist
for name in top_genre:
    if os.path.exists(os.path.join(images_dir, name)):
        print("Found -->", os.path.join(images_dir, name))
    else:
        print("Did not find -->", os.path.join(images_dir, name))
        
# Augment data
batch_size = 16
train_input_shape = (224, 224, 3)
n_classes = genre_top.shape[0]

train_datagen = ImageDataGenerator(validation_split=0.2,
                                    rescale=1./255.,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=10,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True
                                )

train_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="training",
                                                    shuffle=True,
                                                    classes=top_genre.tolist()
                                                   )

valid_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="validation",
                                                    shuffle=True,
                                                    classes=top_genre.tolist()
                                                   )

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)

# Print a random paintings and it's random augmented version
fig, axes = plt.subplots(1, 2, figsize=(20,10))

random_genre = random.choice(top_genre)
random_image = random.choice(os.listdir(os.path.join(images_dir, random_genre)))
random_image_file = os.path.join(images_dir, random_genre, random_image)

# Original image
image = plt.imread(random_image_file)
axes[0].imshow(image)
axes[0].set_title("An original Image of " + random_genre)
axes[0].axis('off')

# Transformed image
aug_image = train_datagen.random_transform(image)
axes[1].imshow(aug_image)
axes[1].set_title("A transformed Image of " + random_genre)
axes[1].axis('off')

plt.show()

# Load pre-trained model
def CNN_model(base):
    if base == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)
    elif base == 'resnet101':
        base_model = ResNet101(weights='imagenet', include_top=False, input_shape=train_input_shape)
    elif base == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=train_input_shape)
    elif base == 'inceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=train_input_shape)

    for layer in base_model.layers:
        layer.trainable = True
    
    return base_model
    
# Add layers at the end
base_model = CNN_model('inceptionV3')
X = base_model.output
X = Flatten()(X)

X = Dense(512, kernel_initializer='he_uniform')(X)
X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

X = Dense(16, kernel_initializer='he_uniform')(X)
X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

output = Dense(n_classes, activation='softmax')(X)

model = Model(inputs=base_model.input, outputs=output)

optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])
              
n_epoch = 50

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')
                              
# Train the model - all layers
history1 = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=n_epoch,
                              shuffle=True,
                              verbose=1,
                              callbacks=[reduce_lr],
                              #use_multiprocessing=True,
                              #workers=16,
                              class_weight=class_weights
                             )
                             
history = {}
history['loss'] = history1.history['loss']
history['acc'] = history1.history['accuracy']
history['val_loss'] = history1.history['val_loss']
history['val_acc'] = history1.history['val_accuracy']
history['lr'] = history1.history['lr']

# Plot the training graph
def plot_training(history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    
    axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')
    axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend(loc='best')

    axes[1].plot(epochs, loss, 'r-', label='Training Loss')
    axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend(loc='best')
    
    plt.show()
    
plot_training(history)

def showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID):
    # Loop on each generator batch and predict
    y_pred, y_true = [], []
    for i in range(STEP_SIZE_VALID):
        (X,y) = next(valid_generator)
        y_pred.append(model.predict(X))
        y_true.append(y)
    
    # Create a flat list for y_true and y_pred
    y_pred = [subresult for result in y_pred for subresult in result]
    y_true = [subresult for result in y_true for subresult in result]
    
    # Update Truth vector based on argmax
    y_true = np.argmax(y_true, axis=1)
    y_true = np.asarray(y_true).ravel()
    
    # Update Prediction vector based on argmax
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.asarray(y_pred).ravel()
    
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(10,10))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    conf_matrix = conf_matrix/np.sum(conf_matrix, axis=1)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", square=True, cbar=False, 
                cmap=plt.cm.jet, xticklabels=tick_labels, yticklabels=tick_labels,
                ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.show()
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=np.arange(n_classes), target_names=top_genre.tolist()))
    
# Classification report and confusion matrix
tick_labels = top_genre.tolist()

showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID)