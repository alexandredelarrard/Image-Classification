# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:07:13 2017

@author: alexandre
"""

import numpy as np
import cv2
import pandas as pd
import os
import json

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers


class Train_keras(object):

    def __init__(self, global_parameters, pictures_dict, params_cnn):

        im_list_path = pictures_dict["pictures_train_classification"]

        self.batch_size         = params_cnn["batch_size"]
        self.image_size         = params_cnn["image_size"]
        self.epoch_size         = params_cnn["epoch_size"]
        self.keep_probability   = params_cnn["keep_probability"]
        self.weight_decay       = params_cnn["weight_decay"]
        self.optimizer          = params_cnn["optimizer"]
        self.learning_rate      = params_cnn["learning_rate"]
        self.max_nrof_epochs    = params_cnn["num_epoch"]
        self.patience           = params_cnn["patience"]

        self.seed               = params_cnn["seed"]
        self.data_augmentation  = params_cnn["data_augmentation"]
        self.pretrained_model   = None
        self.proportion_train_test = params_cnn["split_train_test"]

        self.path_client        =  "/".join([global_parameters["ibm_client_train_data_path_classification"], global_parameters["fam_id"]]) + "/"
        self.model_dir          =  global_parameters['path_%s_%s_%s']

        self.Main(im_list_path)


    def Main(self, im_list_path):

        liste = pd.DataFrame(im_list_path)[0].apply(lambda x: os.path.dirname(x).replace(self.path_client, ""))

        label_map = {}
        for j, classe in enumerate(liste.unique()):
            label_map[classe] = j

        x_train, y_train = self.load_data(im_list_path, label_map)

        weights= pd.DataFrame(np.argmax(y_train, 1))[0].value_counts()
        dict_weights = {}
        max_it       = weights[weights.index.tolist()[0]]
        for i in weights.index.tolist():
            dict_weights[i] = max_it/float(weights[i])

        #### train model keras 5 conv + 3 dense
        model = self.Keras_model(y_train, x_train, self.image_size, label_map, dict_weights)

        #serialize model to JSON
        model_json = model.to_json()
        with open(self.model_dir + "/model.json", "w") as json_file:
            json_file.write(model_json)

        with open(self.model_dir + "/classes.json", "w") as json_file:
            json.dump(label_map, json_file)


    def setup_generator(self, X_train, X_valid, Y_train, Y_valid, batch_size=32):

        train_datagen = ImageDataGenerator(
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    rotation_range=7,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zca_whitening=False,
                                    fill_mode='reflect')

        train_datagen.fit(X_train)

        test_datagen = ImageDataGenerator(zoom_range=0.01)
        test_datagen.fit(X_valid)

        X_train_aug = train_datagen.flow(X_train, Y_train, seed=0, batch_size=batch_size)
        X_test_aug = test_datagen.flow(X_valid, Y_valid, seed=0, batch_size=batch_size)

        return X_train_aug, X_test_aug


    def Keras_model(self, y_train, x_train, image_resize, labels, dict_weights):

        X_rand = np.random.uniform(0,1,len(x_train))
        test_index  = X_rand <= self.proportion_train_test
        train_index = X_rand  >  self.proportion_train_test

        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[test_index]
        Y_valid = y_train[test_index]

        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        kfold_weights_path = self.model_dir + "/weights_kfold.h5"

        model = Sequential()
        model.add(Conv2D(32, 3, 3, init="he_normal",  activation='relu', input_shape=(image_resize, image_resize, 3))) ### 64*64 ----> 32*32*32
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, 3, 3, init="he_normal",  activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, 3, 3, init="he_normal",  activation='relu')) 
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, 3, 3, init="he_normal", activation='relu')) 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, 3, 3, init="he_normal", activation='relu')) 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(1024, init="he_normal", activation='relu')) #### FC 1024
        model.add(Dropout(1- self.keep_probability))

        model.add(Dense(1024, init="he_normal", activation='relu')) #### FC 1024
        model.add(Dropout(1- self.keep_probability))

        model.add(Dense(len(labels.keys()), activation='softmax'))

        rms = optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08, decay=self.weight_decay)
        model.compile(loss='binary_crossentropy',
                      optimizer=rms,
                      metrics= ['accuracy'])

        callbacks = [
                EarlyStopping(monitor='val_loss', patience=11, verbose=0),
                ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

        if not self.data_augmentation:
            model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),
                          batch_size=self.batch_size, verbose=2, nb_epoch=300, callbacks=callbacks,
                          shuffle=True, class_weight=dict_weights)

        else:
            X_train_aug, X_test_aug = self.setup_generator(X_train, X_valid, Y_train, Y_valid, batch_size=self.batch_size)

            model.fit_generator(X_train_aug, samples_per_epoch= self.epoch_size, nb_epoch=self.max_nrof_epochs,
                                validation_data=X_test_aug, nb_val_samples=len(X_valid), class_weight=dict_weights, callbacks=callbacks)

        p_valid = model.predict(X_valid, batch_size = self.batch_size, verbose=2)
        print(p_valid)
        print("True accuracy is : %s"%np.mean(np.equal(np.argmax(p_valid,1), np.argmax(Y_valid,1))))

        return model


    def load_data(self, image_paths, label_map):

            nrof_samples = len(image_paths)
            print("number images size %s" %str(nrof_samples))
            img_list = [None] * nrof_samples
            y_list = []

            for i in range(nrof_samples):
                try:
                    img=self.image_descriptors(image_paths[i])
                    img_list.append(self.prewhiten(img))
                    targets = np.zeros(len(label_map.keys()))
                    targets[label_map[os.path.dirname(image_paths[i]).replace(self.path_client, '')]] = 1
                    y_list.append(targets)
                    if i%500 ==0:
                       print("image  %i/%i with shape %s"%(i, nrof_samples, img_list[i].shape))

                except Exception, e:
                    print("could not open picture %s"%image_paths[i])
                    print(e)
                    pass

            return np.array(img_list), np.array(y_list)


    def image_descriptors(self, X):
        image= cv2.imread(X)
        image= cv2.resize(image, (self.image_size, self.image_size))
        return image

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)

        return y
