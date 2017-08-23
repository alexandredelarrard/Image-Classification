# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:05:03 2017

@author: alexandre
"""

import numpy as np
import time
import cv2
import json
import sys, os
from keras.models import model_from_json

sys.path.append("/".join([os.environ["Q_PATH"], "Qopius_back_V2", "client"]))
from client_utils.wrapper_dynamoDB    import WrapperDynamoDB


class Test_keras(object):

    def __init__(self, pictures_dict, global_parameters, results, parameters_models):
        
        self.parameters_keras_models   = parameters_models
        self.global_parameters         = global_parameters
        self.results                   = results

        self.keras_classes = self.global_parameters["model_path"] +  '/classes.json'
        self.keras_weights = self.global_parameters["model_path"] +  '/weights_kfold.h5'
        self.keras_model   = self.global_parameters["model_path"] +  '/model.json'

        self.UPC_recognition(pictures_dict, self.results)


    def UPC_recognition(self, pictures_dict, results):

         total = time.time()
         imagette_information = {"pil_images" : [],
                                 "images_descriptions" : {"box_id": [],
                                                          "image_name" : []},
                                 "images_features" : []}
                                 
#         if len(self.results) ==0:
#             for test_image_name in pictures_dict.keys():
#                self.check_results(test_image_name)
#                
#                if parents_ID_0 not in self.results[test_image_name].keys():
#                    self.results[test_image_name][parents_ID_0] = {}

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
         
         ###########  shape the result dictionnary  ##########
         if len(imagette_information["pil_images"])>0:
             self.Searcher_keras(imagette_information, parents_ID_0)
             
         print("total test time pred for keras model : %s" %str(time.time() - total))


    def Searcher_keras(self, images_dict, type_reco):

        #load the encoder of classes
        with open(self.keras_classes, 'r') as f:
            loaded_classes = json.load(f)
            
        if len(str(loaded_classes.keys()[0])) > 1:
            loaded_classes = {str(v): k for k, v in loaded_classes.iteritems()}
            
        levels_dict = {}
        for k, v in loaded_classes.items():
            v = str(v.replace('\\', "/").split("/")[0])
            if v not in levels_dict.keys():
                levels_dict.update(self.getLevelOfLabelID(v))

        json_file    = open(self.keras_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.keras_weights)
        
        with open(self.keras_model, 'r') as files:
            model_json = json.load(files)
            
        self.image_size = int(model_json['config'][0]['config']['batch_input_shape'][1])
        images_test = self.load_data(images_dict["pil_images"])
        
        predictions = loaded_model.predict(images_test, batch_size = 128, verbose=2)

        for p in range(len(predictions)):
            image_name = images_dict["images_descriptions"]["image_name"][p]
            box_id     = images_dict["images_descriptions"]["box_id"][p]

            sorted_index = np.argsort(predictions[p])[::-1]
            upc_results = list(sorted_index[: self.parameters_keras_models["nb_top_upc"]])
            
            for k, pred_k in enumerate(upc_results):
                answer = loaded_classes[str(pred_k)].replace('\\', "/").split("/")

                #### si "candidates" n'est pas une clé du dico de box_id  alors on le crée
                if "candidates" not in self.results[image_name][type_reco][box_id].keys():
                    self.results[image_name][type_reco][box_id]["candidates"] = {}

                self.results[image_name][type_reco][box_id]["candidates"][str(k)] = {}
                
                ### fill in highest levels for the highest level of each prediction prediction
                nbr_parents_levels = len(levels_dict[answer[0]])
                if nbr_parents_levels>0:
                    for parents_level, parents_value in levels_dict[answer[0]].items():
                        self.results[image_name][type_reco][box_id]["candidates"][str(k)][parents_level] = parents_value

                for level in range(len(answer)) :
                    if "no_label" not in answer[level] and float(np.max(predictions[p])) > self.parameters_keras_models["threshold_no_label"]:
                        self.results[image_name][type_reco][box_id]["candidates"][str(k)]['level_%s'%str(level + nbr_parents_levels)] = int(answer[level])
                
            self.results[image_name][type_reco][box_id]['probability'] = float(np.max(predictions[p]))
            

    def load_data(self, images):

            nrof_samples = len(images)
            print("number images size %s" %str(nrof_samples))
            img_list = [None] * nrof_samples

            for i in range(len(images)):
                img = self.image_descriptors(images[i])
                img_list[i] = self.prewhiten(img)

            return np.stack(img_list)


    def image_descriptors(self, X):
        image= cv2.resize(np.array(X), (self.image_size, self.image_size))
        return image


    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)

        return y
        
        
    def getLevelOfLabelID(self, labelID):
        """
        Gets the parent labels of label
        @labelID  <str>
        """
        # fetch item
        try:
            labelItem = WrapperDynamoDB().get_item({"id": str(labelID)}, "label")
        except Exception:
            labelItem = WrapperDynamoDB().get_item({"v1": str(labelID)}, "label")
        
        # store all levels
        pl = {str(k): str(v) for k, v in labelItem.items() if k.startswith('level_')}
        
        return {labelID : pl}
        
        
    def check_results(self, new_key):
        if new_key not in self.results.keys():
            self.results[new_key] = {}


