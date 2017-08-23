# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:53:02 2017

@author: alexandre
"""

import tensorflow as tf
import numpy as np
import facenet
import os
import time
import math
import pandas as pd
from sklearn import preprocessing
from PIL import Image
import sys
sys.path.append(os.environ["Q_PATH"] + "/q-engine")
from qopius_visual.utils.utils_facenet.models import inception_resnet_v1

def Filter_data(image_list, path_client, image_dir, image_size, model_file, min_per_class, keep_percentile):

    """
    image_list = path of images to read
    label_list = labels in string e.g: 6500/59566415181 ...
    image_dir = qopius directory to read images
    path_client = clients storage path
    """
    suppress_list = []
    train_set_counter, label_list = get_pairs(image_list, path_client, image_dir)
    batch_size = 600

    le = preprocessing.LabelEncoder()
    le.fit(list(set(label_list)))
    label_list_transformed = le.transform(label_list)

    with tf.Graph().as_default():
        # Get a list of image paths and their labels

        image_batch, label_batch = facenet.read_and_augument_data(image_list, label_list_transformed, image_size, batch_size, None,
            False, False, False, nrof_preprocess_threads=4, shuffle=False)
        prelogits, _ = inception_resnet_v1.inference(image_batch, 1.0,
            phase_train=False, weight_decay=0.0, reuse=False)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        saver = tf.train.Saver(tf.global_variables())
        coord = tf.train.Coordinator()

        config = tf.ConfigProto()
        config.log_device_placement=True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver.restore(sess, os.path.join(os.path.expanduser(model_file)))
            tf.train.start_queue_runners(sess=sess, coord= coord)

            embedding_size = int(embeddings.get_shape()[1])
            nrof_batches = int(math.ceil(len(image_list)/ batch_size))
            nrof_images = len(label_list_transformed)

            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches):
                timess = time.time()
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                emb_array[start_index:end_index,:], idx = sess.run([embeddings, label_batch])
                print("batch %i time : %s"%(i, str(time.time() - timess)))

        for classe in train_set_counter.keys():
            print("class = " + str(classe))

            if len(train_set_counter[classe]) > min_per_class:
                idx = [i for i, e in enumerate(image_list) if e in np.array(train_set_counter[classe])]
                assert len(idx) == len(train_set_counter[classe])
                emb_classe = emb_array[np.array(idx),:]
                center = np.mean(emb_classe, axis=0)
                diffs = emb_classe - center
                dists_sqr = np.sum(np.square(diffs), axis=1)
                threshold = find_threshold(dists_sqr, keep_percentile)
                idx_supp = np.where(np.array(dists_sqr) >= threshold)
                index_in_list = list(np.array(idx)[list(idx_supp[0])])

                for i in list(np.array(image_list)[index_in_list]):
                    suppress_list.append(i)

#                for i, im_path in enumerate(list(np.array(image_list)[index_in_list])):
#                    image= Image.open(im_path)
#                    image.save("/home/ubuntu/Version_automatisation_test/src/image_%s_%i.jpg"%(classe.replace("/","_"),i), "JPEG")


        print("longueur suppress list = %i"%len(suppress_list))
        dataset = [x for x in image_list if x not in suppress_list]

        print("length dataset before : " + str(len(image_list)))
        print("length dataset after remove outliers " + str(len(dataset)))
        return dataset

def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold

def get_pairs(list_path, path_client, image_dir):

         ### Create in the first place a dictionnary with the whole path to a new class and the list of all its pictures (with the whole path before)
         counter_class = {}

         #### delete all paths with augmented pictures or reference pictures (assure that no /images is involved) and 'reference' not in x 'augmented' not in x and
         #### List of pictures for train which have a class
         list_path.sort()
         liste_without_root = pd.DataFrame(list_path)[0].apply(lambda x: os.path.dirname(x))

         if path_client:
             liste_without_root = pd.DataFrame(liste_without_root)[0].apply(lambda x: x.replace(path_client, ""))
         elif image_dir:
             liste_without_root = pd.DataFrame(liste_without_root)[0].apply(lambda x: x.replace(image_dir, ""))
         elif image_dir and path_client:
             liste_without_root = pd.DataFrame(liste_without_root)[0].apply(lambda x: x.replace(path_client, "").replace(image_dir, ""))

         var_path = ''

         for i, path in enumerate(list_path) :
             if var_path ==  liste_without_root[i]:
                counter_class[var_path].append(path)

             else :
                var_path = liste_without_root[i]
                counter_class[var_path] = []

         return counter_class, liste_without_root

