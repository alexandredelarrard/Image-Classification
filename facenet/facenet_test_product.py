# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 10:24:39 2017

@author: alexandre
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 14:35:19 2017

@author: alexandre
"""


import numpy as np
import math
import time
import cv2
import pandas as pd
import scipy
import json

import tensorflow as tf
from   tensorflow.python.ops import data_flow_ops
import tensorflow.contrib.slim as slim

from sklearn import preprocessing
from sklearn.externals import joblib

from facenet_utils.facenet import prewhiten
from facenet_utils.models.inception_resnet_v1 import inference

class Test_product_facenet(object):

    def __init__(self, pictures_dict, global_parameters, results, parameters_facenet_models):
        self.parameters_facenet_models  = parameters_facenet_models
        self.global_parameters         = global_parameters
        self.results                   = results
        
        self.UPC_recognition(pictures_dict)


    def UPC_recognition(self, pictures_dict):

        total = time.time()
        imagette_information = {"pil_images" : [],
                                 "images_descriptions" : {"box_id": [],
                                                          "image_name" : []},
                                 "images_features" : []}

        ###########  create a dictionnary of all pictures (cropped ones) to do image recognition by batch
        image_id = self.results.keys()[0]
        
        if len(self.results[image_id].keys()) >1:
            print("be careful several label_IDs in results, will do classification on the first one: %s"%self.results[image_id].keys())
            parents_ID_0 = self.results[image_id].keys()[0]
            
        if len(self.results[image_id].keys()) ==0:
            print("be careful no libel_ID in the result dictionnary, just classification performed")
            parents_ID_0 = self.global_parameters["required"]["labelID_0"]
            self.results[image_id][parents_ID_0] = {}
            
        if len(self.results[image_id].keys()) ==1:
            parents_ID_0 = self.results[image_id].keys()[0]
         
        for image_id in self.results.keys() :
            result_image =  self.results[image_id][parents_ID_0]
            
            # for each box of each image of the batch
            for box_id, box_info in result_image.iteritems():
                imagette_information["images_descriptions"]["box_id"].append(box_id)
                imagette_information["images_descriptions"]["image_name"].append(str(image_id))
                imagette_information["pil_images"].append(pictures_dict[image_id][result_image[box_id]["y1"]:result_image[box_id]["y2"], result_image[box_id]["x1"]:result_image[box_id]["x2"]])
         
        ########### extract features of length embeddings size ##########
        start = time.time()
        self.Features_extraction(imagette_information)
        print('Time for feature extraction %s' %(time.time()-start))
        
        ###########  shape the result dictionnary  ##########
        start = time.time()
        self.UPC_prediction(imagette_information, parents_ID_0)
        print("total test time pred for tensorflow product: %s" %str(time.time() - total))


    def Features_extraction(self, images_dict):

        images_test, descriptors = self.load_data(images_dict["pil_images"])

        embeddings_size = self.parameters_facenet_models["embedding_size"]
        batch_size      = min(600, len(images_test))
        
        coord = tf.train.Coordinator()

        with tf.Graph().as_default():
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

            input_queue = data_flow_ops.FIFOQueue(capacity=1000,
                                        dtypes=[tf.string, tf.int64],
                                        shapes=[(1,), (1,)],
                                        shared_name=None, name=None)

            nrof_preprocess_threads = 4

            images_and_labels = []
            for _ in range(nrof_preprocess_threads):
                    filenames, label = input_queue.dequeue()
                    images = []
                    
                    for filename in tf.unstack(filenames):

                        file_contents = tf.read_file(filename)
                        image = tf.image.decode_jpeg(file_contents)
                        image = tf.image.resize_images(image, [self.parameters_facenet_models["image_size"], self.parameters_facenet_models["image_size"]])

                        image.set_shape((self.parameters_facenet_models["image_size"], self.parameters_facenet_models["image_size"], 3))
                        images.append(tf.image.per_image_standardization(image))

                    images_and_labels.append([images, label])

            image_batch, labels_batch = tf.train.batch_join(
                images_and_labels, batch_size=batch_size_placeholder,
                shapes=[(self.parameters_facenet_models["image_size"], self.parameters_facenet_models["image_size"], 3), ()], enqueue_many=True,
                capacity=4 * nrof_preprocess_threads * batch_size,
                allow_smaller_final_batch=True)

            image_batch = tf.identity(image_batch, 'image_batch')

            batch_norm_params ={
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # force in-place updates of mean and variance estimates
            'updates_collections': None,
            # Moving averages ends up in the trainable variables collection
            'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
            }

            # Build the inference graph
            prelogits, _ =  inference(image_batch, 1.0, phase_train=False, weight_decay=0.)

            bottleneck = slim.fully_connected(prelogits, embeddings_size, activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    weights_regularizer=slim.l2_regularizer(0.),
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params,
                    scope='Bottleneck', reuse=False)

            embeddings = tf.nn.l2_normalize(bottleneck, 1, 1e-10, name='embeddings')

            # Start running operations on the Graph.
            config = tf.ConfigProto(allow_soft_placement=True)
            config.log_device_placement = True
            config.gpu_options.allow_growth = True
            coord = tf.train.Coordinator()

            with tf.Session(config = config) as sess:
    
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess=sess, coord = coord)
    
                print(self.global_parameters["model_path"] + "/model-facenet.ckpt")
                tf.train.Saver().restore(sess, self.global_parameters["model_path"] + "/model-facenet.ckpt")
    
                nrof_images = len(images_test)
                nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
                images_dict["images_features"] = np.zeros((nrof_images, embeddings_size))
    
                for i in range(nrof_batches):
                    start_index = i*batch_size
                    end_index = min((i+1)*batch_size, nrof_images)
                    batch_images= images_test[start_index:end_index, :, :, :]
                    images_dict["images_features"][start_index:end_index,:] = sess.run(embeddings , {"image_batch:0": batch_images})

        images_dict["images_features"] = np.concatenate((images_dict["images_features"], descriptors), axis=1) # ----> when train will be done with descriptors


    def UPC_prediction(self, images_dict, type_reco):

        #load the encoder of classes
        liste_labels = pd.read_csv(self.global_parameters["model_path"] + "/classes.csv", sep= ",", header = 0)
        liste_labels = list(liste_labels[liste_labels.columns[0]])
        encoder = preprocessing.LabelEncoder()
        encoder.classes_ = liste_labels

        ## use fast sgd classifier to have prediction probabilities
        clf = joblib.load(self.global_parameters["model_path"] + "/index_shelf_svc.pkl")
        prb = joblib.load(self.global_parameters["model_path"] + "/index_shelf_sgd.pkl") 

        predictions       =  clf.predict(images_dict["images_features"])
        predictions_proba =  prb.predict_proba(images_dict["images_features"])

        for p in range(len(images_dict["images_features"])):
            image_name = images_dict["images_descriptions"]["image_name"][p]
            box_id     = images_dict["images_descriptions"]["box_id"][p]
            
            sorted_index = np.argsort(predictions_proba[p])[::-1]
            upc_results = [predictions[p]] + list(sorted_index[1: self.parameters_facenet_models["nb_top_upc"]])

            for k, pred_k in enumerate(upc_results):
                answer = encoder.inverse_transform(pred_k).split("/")
                
                #### si "candidates" n'est pas une clé du dico de box_id  alors on le crée
                if "candidates" not in self.results[image_name][type_reco][box_id].keys():
                    self.results[image_name][type_reco][box_id]["candidates"] = {}

                self.results[image_name][type_reco][box_id]["candidates"][str(k)] = {}
                
                for level in range(len(answer)) :
                    if "no_label" in answer[level] :
                        for under_levels in range(level, len(answer)):
                            self.results[image_name][type_reco][box_id]["candidates"][str(k)]['level_%s'%str(under_levels)] = None
                    else:
                        self.results[image_name][type_reco][box_id]["candidates"][str(k)]['level_%s'%str(level)] = int(answer[level])

            self.results[image_name][type_reco][box_id]['probability'] = float(np.max(predictions_proba[p]))
            
        
        
    def load_data(self, images):

            nrof_samples = len(images)
            print("number images size %s" %str(nrof_samples))
            img_list = [None] * nrof_samples
            descriptors = np.zeros((nrof_samples, 19))
            
            for i in range(len(images)):
                img, descriptors[i,:] = self.image_descriptors(images[i]) 
                img_list[i] = prewhiten(img)

            return np.stack(img_list), descriptors
            
            
    def image_descriptors(self, X):
        
        image= cv2.resize(X, (self.parameters_facenet_models["image_size"], self.parameters_facenet_models["image_size"]))
        
        ratio = image.shape[0]/float(image.shape[1])   

        means = [np.mean(image[:,:,0].ravel()), np.mean(image[:,:,1].ravel()), np.mean(image[:,:,2].ravel())]
        stds  = [np.std(image[:,:,0].ravel()), np.std(image[:,:,1].ravel()), np.std(image[:,:,2].ravel())]
         
        mins = [image[:,:,0].min(), image[:,:,1].min(), image[:,:,2].min()]
        maxs = [image[:,:,0].max(), image[:,:,1].max(), image[:,:,2].max()]
        
        kurt = [scipy.stats.kurtosis(image[:,:,0].ravel()), scipy.stats.kurtosis(image[:,:,1].ravel()), scipy.stats.kurtosis(image[:,:,2].ravel())]
        skew = [scipy.stats.skew(image[:,:,0].ravel()), scipy.stats.skew(image[:,:,1].ravel()), scipy.stats.skew(image[:,:,2].ravel())]
        
        out = means + stds + mins + maxs + kurt + skew + [ratio]

        return image, np.array(out)*0.5
