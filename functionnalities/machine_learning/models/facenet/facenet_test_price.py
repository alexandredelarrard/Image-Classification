# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:55:25 2017

@author: alexandre
"""
import numpy as np
import math
import time
import cv2
import pandas as pd

import tensorflow as tf
from   tensorflow.python.ops import data_flow_ops

from facenet_utils.facenet import prewhiten
from facenet_utils import facenet
from facenet_utils.models.inception_resnet_v1 import inference
from facenet_utils.price_utils import set_commat_to_price

class Test_price_facenet(object):

    def __init__(self, pictures_dict, global_parameters, results, parameters_facenet_models):
        self.parameters_facenet_models  = parameters_facenet_models
        self.global_parameters         = global_parameters
        self.results                   = results
        
        self.UPC_recognition(pictures_dict, self.results)


    def UPC_recognition(self, pictures_dict, results):

        total = time.time()
        imagette_information = {"pil_images" : [],
                                 "images_descriptions" : {"box_id": [],
                                                          "image_name" : []},
                                 "images_features" : []}

        ###########  create a dictionnary of all pictures (cropped ones) to do image recognition by batch
        image_id = results.keys()[0]
        
        if len(results[image_id].keys()) >1:
            print("be careful several label_IDs in results, will do classification on the first one: %s"%results[image_id].keys())
            parents_ID_0 = results[image_id].keys()[0]
            
        if len(results[image_id].keys()) ==0:
            print("be careful no libel_ID in the result dictionnary, just classification performed")
            parents_ID_0 = self.global_parameters["required"]["labelID_0"]
            results[image_id][parents_ID_0] = {}
            
        if len(results[image_id].keys()) ==1:
            parents_ID_0 = results[image_id].keys()[0]
         
        for image_id in results.keys() :
            result_image =  results[image_id][parents_ID_0]
            
            # for each box of each image of the batch
            for box_id, box_info in result_image.iteritems():
                imagette_information["images_descriptions"]["box_id"].append(box_id)
                imagette_information["images_descriptions"]["image_name"].append(str(image_id))
                imagette_information["pil_images"].append(pictures_dict[image_id][result_image[box_id]["y1"]:result_image[box_id]["y2"], result_image[box_id]["x1"]:result_image[box_id]["x2"]])
                 
        ########### extract features of length embeddings size ##########
        if len(imagette_information["pil_images"])>0:
             start = time.time()
             self.Features_extraction(imagette_information)
             print('Time for feature extraction %s' %(time.time()-start))
        
             ###########  shape the result dictionnary  ##########
             start = time.time()
             self.price_prediction(imagette_information, parents_ID_0)
             print("total test time pred for tensorflow price: %s" %str(time.time() - total))
          
         #### If no prices are detected, we pop out the price key  
        else:
            print("No price images to detect as no boxe was detected at all")
#                for key in self.results.keys():
#                    self.results[key].pop(parents_ID_0) 
            

    def price_prediction(self, imagette_information, type_reco):
        
        imagette_information["images_features"] = pd.DataFrame(imagette_information["images_features"]).astype(int).astype(str)
        imagette_information["images_features"][imagette_information["images_features"] == '10'] = ''
        
        X = imagette_information["images_features"][0]
        for i in range(1,6):
            X += imagette_information["images_features"][i]
            
        X = set_commat_to_price(X)

        for p in range(len(imagette_information["images_features"])):
            image_name = imagette_information["images_descriptions"]["image_name"][p]
            box_id     = imagette_information["images_descriptions"]["box_id"][p]
            
            self.results[image_name][type_reco][box_id]["value"] = X[p]


    def Features_extraction(self, images_dict):

        images_test = self.load_data(images_dict["pil_images"])

        batch_size      = min(600, len(images_test))
        num_labels      = 11
        
        coord = tf.train.Coordinator()

        with tf.Graph().as_default():
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

            input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                            dtypes=[tf.string, tf.int32, tf.int32],
                                            shapes=[(1,), (1,), (6,)],
                                            shared_name=None, name=None)
    
            nrof_preprocess_threads = 4
            images_and_labels = []
            for _ in range(nrof_preprocess_threads):
                filenames, label, coordinates = input_queue.dequeue()
                images = []
                for filename in tf.unstack(filenames):
    
                    file_contents = tf.read_file(filename)
                    image = tf.image.decode_jpeg(file_contents, fancy_upscaling=True)
                    image = tf.image.resize_images(image, [self.parameters_facenet_models["image_size"], self.parameters_facenet_models["image_size"]])
    
                    image.set_shape((self.parameters_facenet_models["image_size"], self.parameters_facenet_models["image_size"], 3))
                    images.append(tf.image.per_image_standardization(image))
    
                images_and_labels.append([images, label, [coordinates]])
    
            image_batch, labels_batch, labels_number_batch = tf.train.batch_join(
                images_and_labels, batch_size=batch_size_placeholder,
                shapes=[(self.parameters_facenet_models["image_size"], self.parameters_facenet_models["image_size"], 3), (), (6,)], enqueue_many=True,
                capacity= 4 * nrof_preprocess_threads * batch_size,
                allow_smaller_final_batch=True)
    
            image_batch = tf.identity(image_batch, 'image_batch')
    
            prelogits, end_points = inference(image_batch, 1., phase_train=False, weight_decay=0.)
            bottleneck = end_points['prelogits'].get_shape().as_list()[1]
    
            # Softmax 1
            w_s1 = facenet.weight_variable([bottleneck, num_labels], 'w_s1')
            b_s1 = facenet.bias_variable([num_labels], 'b_s1')
    
            # Softmax 2
            w_s2 = facenet.weight_variable([bottleneck, num_labels], 'w_s2')
            b_s2 =  facenet.bias_variable([num_labels], 'b_s2')
    
            # Softmax 3
            w_s3 = facenet.weight_variable([bottleneck, num_labels], 'w_s3')
            b_s3 =  facenet.bias_variable([num_labels], 'b_s3')
    
            # Softmax 4
            w_s4 = facenet.weight_variable([bottleneck, num_labels], 'w_s4')
            b_s4 =  facenet.bias_variable([num_labels], 'b_s4')
    
            # Softmax 5
            w_s5 = facenet.weight_variable([bottleneck, num_labels], 'w_s5')
            b_s5 =  facenet.bias_variable([num_labels], 'b_s5')
    
            # Softmax 6
            w_s6 = facenet.weight_variable([bottleneck, num_labels], 'w_s6')
            b_s6 =  facenet.bias_variable([num_labels], 'b_s6')
    
            with tf.name_scope("softmax_1"):
                logits_1 = tf.nn.softmax(tf.matmul(prelogits, w_s1) + b_s1)
            with tf.name_scope("softmax_2"):
                logits_2 = tf.nn.softmax(tf.matmul(prelogits, w_s2) + b_s2)
            with tf.name_scope("softmax_3"):
                logits_3 = tf.nn.softmax(tf.matmul(prelogits, w_s3) + b_s3)
            with tf.name_scope("softmax_4"):
                logits_4 = tf.nn.softmax(tf.matmul(prelogits, w_s4) + b_s4)
            with tf.name_scope("softmax_5"):
                logits_5 = tf.nn.softmax(tf.matmul(prelogits, w_s5) + b_s5)
            with tf.name_scope("softmax_6"):
                logits_6 = tf.nn.softmax(tf.matmul(prelogits, w_s6) + b_s6)
    
            prediction = tf.stack([tf.cast(tf.argmax(logits_1, 1), tf.int32), tf.cast(tf.argmax(logits_2, 1), tf.int32), tf.cast(tf.argmax(logits_3, 1), tf.int32),
                                  tf.cast(tf.argmax(logits_4, 1), tf.int32), tf.cast(tf.argmax(logits_5, 1), tf.int32), tf.cast(tf.argmax(logits_6, 1), tf.int32)], axis=1)
    
            with tf.Session() as sess:
    
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess=sess, coord = coord)
                tf.train.Saver().restore(sess, self.global_parameters["model_path"] + "/model-facenet.ckpt")
    
                nrof_images = len(images_test)
                nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
                images_dict["images_features"] = np.zeros((nrof_images, 6))
    
                for i in range(nrof_batches):
                    start_index = i*batch_size
                    end_index = min((i+1)*batch_size, nrof_images)
                    batch_images= images_test[start_index:end_index, :, :, :]
                    images_dict["images_features"][start_index:end_index,:] = sess.run(prediction, {"image_batch:0": batch_images})
        
        return images_dict
        
        
    def load_data(self, images):

        nrof_samples = len(images)
        print("number images size %s" %str(nrof_samples))
        img_list = [None] * nrof_samples
        
        for i in range(len(images)):
            img = self.image_descriptors(images[i]) 
            img_list[i] = prewhiten(img)

        return np.stack(img_list)
            
            
    def image_descriptors(self, X):
        image= cv2.resize(X, (self.parameters_facenet_models["image_size"], self.parameters_facenet_models["image_size"]))
        return image
