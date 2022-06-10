'''
script to train early_model
'''

# System imports

# 3rd party imports
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import tensorflow_hub as hub
import pandas as pd
import cv2

import matplotlib.pyplot as plt

# local imports
from utils import Data

# end file header
__author__      = 'Adrian Lubitz'

## Load Model
early_model = keras.models.load_model('early_fusion_weights_0.92.hdf5', custom_objects={'KerasLayer': hub.KerasLayer})
early_model.load_weights('early_fusion_weights_0.92.hdf5')
sgd = optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=False)
early_model.compile(loss='categorical_crossentropy', 
              optimizer=sgd, 
              metrics=['accuracy'])


## Load Data
d = Data(load='both')
print(d.train.shape)
d.train.head()
data_test = d.tf_data('images/test/*/*.jpg')
data_train = d.tf_data('images/train/*/*.jpg')

print(d.test.shape)
d.test.head()    


## Train Model
# Setup callbacks, logs and early stopping condition
checkpoint_path = "stacking_early_fusion/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
cp = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy',save_best_only=True,verbose=1, mode='max')
csv_logger = callbacks.CSVLogger('stacking_early_fusion/stacking_early.log')
es = callbacks.EarlyStopping(patience = 3, restore_best_weights=True)

# Reduce learning rate if no improvement is observed
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.1, patience=1, min_lr=0.00001)

# Training
history = early_model.fit(data_train,
                   epochs=15,
                   steps_per_epoch = d.train.shape[0]//d.batch_size,
                   validation_data = data_test,
                   validation_steps = d.test.shape[0]//d.batch_size,
                   callbacks=[cp, csv_logger, reduce_lr])

                   