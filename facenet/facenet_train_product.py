# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 10:24:47 2017

@author: alexandre
"""

import os.path
import time
import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim
import logging
from tensorflow.python.ops import data_flow_ops
import math
import cv2
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from sklearn import linear_model
from sklearn.svm import LinearSVC
import json
from tensorflow.contrib.tensorboard.plugins import projector
import scipy.misc

from facenet_utils.models import inception_resnet_v1
from facenet_utils import facenet


class Train_product_facenet(object):

    def __init__(self, global_parameters, pictures_dict, params_cnn):
        
        self.batch_size         = params_cnn["batch_size"]
        self.image_size         = params_cnn["image_size"]
        self.epoch_size         = params_cnn["epoch_size"]
        self.alpha              = params_cnn["alpha"]
        self.embedding_size     = params_cnn["embedding_size"]
        self.keep_probability   = params_cnn["keep_probability"]
        self.weight_decay       = params_cnn["weight_decay"]
        self.optimizer          = params_cnn["optimizer"]
        self.learning_rate      = params_cnn["learning_rate"]
        self.learning_rate_decay_epochs   = params_cnn["learning_rate_decay_epochs"]
        self.learning_rate_decay_factor   = params_cnn["learning_rate_decay_factor"]
        self.moving_average_decay         = params_cnn["moving_average_decay"]
        self.brand_loss_factor  = params_cnn["brand_loss_factor"]
        self.center_loss_factor = params_cnn["center_loss_factor"]
        self.center_loss_alfa   = params_cnn["center_loss_alfa"]
        self.max_nrof_epochs    = params_cnn["num_epoch"]

        self.seed               = params_cnn["seed"]
        self.pretrained_model   = None
        self.proportion_train_test = params_cnn["split_train_test"]
        
        self.path_client        =  "/".join([global_parameters["data_path"], global_parameters["required"]["labelID_0"]]) + "/"
        self.model_dir          =  global_parameters['model_path']
        self.summary_dir        =  os.path.join("/".join([global_parameters['model_path'], "tmp", "summary_dashboard"]), time.strftime("%H:%M:%S"))
        self.global_parameters  =  global_parameters

        self.Main(pictures_dict)


    def Main(self, im_list_path):

        # Get the paths for the corresponding images
        np.random.seed(seed=self.seed)
        np.random.shuffle(im_list_path)

        ### split paths into train and test
        ### train must not have any augmented image present in test

        self.images_train, self.descriptors_train, self.im_list_train = self.load_data(im_list_path[int(self.proportion_train_test*len(im_list_path)):])
        self.images_test, self.descriptors_test, self.im_list_test = self.load_data(im_list_path[0:int(self.proportion_train_test*len(im_list_path))])

        liste_labels_upc_string = pd.DataFrame(self.im_list_train + self.im_list_test)[0].apply(lambda x : os.path.dirname(x).replace(self.path_client, ""))
        liste_labels_brand_string = pd.DataFrame(self.im_list_train + self.im_list_test)[0].apply(lambda x : os.path.dirname(x).replace(self.path_client, "").split("/")[0])

        self.le = preprocessing.LabelEncoder()
        self.le.fit(list(set(liste_labels_upc_string)))
        pd.DataFrame(self.le.classes_).to_csv(self.model_dir + "/classes.csv", index= False, sep=",")

        self.brand = preprocessing.LabelEncoder()
        self.brand.fit(list(set(liste_labels_brand_string)))

        liste_labels_upc_string_train = pd.DataFrame(self.im_list_train)[0].apply(lambda x : os.path.dirname(x).replace(self.path_client, ""))
        liste_labels_upc_string_test = pd.DataFrame(self.im_list_test)[0].apply(lambda x : os.path.dirname(x).replace(self.path_client, ""))

        liste_labels_brand_string_train = pd.DataFrame(self.im_list_train)[0].apply(lambda x : os.path.dirname(x).replace(self.path_client, "").split("/")[0])
        liste_labels_brand_string_test = pd.DataFrame(self.im_list_test)[0].apply(lambda x : os.path.dirname(x).replace(self.path_client, "").split("/")[0])

        ###  create list of labels
        self.liste_labels_upc_encoded_train = self.le.transform(liste_labels_upc_string_train)
        self.liste_labels_upc_encoded_test = self.le.transform(liste_labels_upc_string_test)

        self.liste_labels_brand_encoded_train = self.brand.transform(liste_labels_brand_string_train)
        self.liste_labels_brand_encoded_test = self.brand.transform(liste_labels_brand_string_test)

        ### create dataset for train
        nrof_classes = len(self.le.classes_)
        nrof_brands = len(self.brand.classes_)

        print("longueur images train : %s"%str(len(self.im_list_train)))
        print("longueur labels upc train %s"%str(len(self.liste_labels_upc_encoded_train)))
        print("longueur labels brand train %s"%str(len(self.liste_labels_brand_encoded_train)))
        print("nombre de classes %s " %str(nrof_classes))
        print("nombre brands %s"%str(nrof_brands))

        with tf.Graph().as_default():
            tf.set_random_seed(self.seed)
            global_step = tf.Variable(0, trainable=False)

            # Create a queue that produces indices into the image_list and label_list
            labels = ops.convert_to_tensor(self.liste_labels_upc_encoded_train, dtype=tf.int32)
            range_size = array_ops.shape(labels)[0]
            index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=self.seed, capacity=32)

            index_dequeue_op = index_queue.dequeue_many(self.batch_size*self.epoch_size, 'index_dequeue')
            
            learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
            labels_brand_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels_brand')
            
            embedding_var = tf.placeholder(tf.float32, shape=(None, self.embedding_size), name='Embedding_bottleneck')

            input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                        dtypes=[tf.string, tf.int64, tf.int64],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)

            enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, labels_brand_placeholder], name='enqueue_op')

            nrof_preprocess_threads = 4
            images_and_labels = []
            for _ in range(nrof_preprocess_threads):
                filenames, label, label_brand_queue = input_queue.dequeue()
                images = []
                for filename in tf.unstack(filenames):

                    file_contents = tf.read_file(filename)
                    image = tf.image.decode_jpeg(file_contents, fancy_upscaling=True)
                    image = tf.image.resize_images(image, [self.image_size + 25, self.image_size + 25])

                    descriptors = tf.py_func(self.image_descriptors, [filenames, "train"], tf.float64)
                    image = tf.random_crop(image, [self.image_size, self.image_size, 3])

                    ### DATA AUGMENTATION
                    if np.random.randint(0,10) >= 4:
                        image = tf.py_func(facenet.random_rotate_image, [image, 10.0], tf.uint8)

                    if np.random.randint(0,10)>= 7:
                        image = tf.image.random_brightness(image, max_delta=64)

                    if np.random.randint(0,10)>= 7:
                        image = tf.image.random_contrast(image, lower=0.5, upper=1.1)

                    image.set_shape((self.image_size, self.image_size, 3))
                    images.append(tf.image.per_image_standardization(image))

                images_and_labels.append([images, label, label_brand_queue, [descriptors]])

            image_batch, labels_batch, labels_brand_batch, descriptors_batch = tf.train.batch_join(
                images_and_labels, batch_size=batch_size_placeholder,
                shapes=[(self.image_size, self.image_size, 3), (), (), (19,)], enqueue_many=True,
                capacity= 4 * nrof_preprocess_threads * self.batch_size,
                allow_smaller_final_batch=True)

            descriptors_batch = tf.identity(descriptors_batch, "descriptors_batch")
            label_batch = tf.identity(labels_batch, 'label_batch')
            labels_brand_batch = tf.identity(labels_brand_batch, 'labels_brand_batch')
            image_batch = tf.identity(image_batch, 'image_batch')

            batch_norm_params_cnn ={
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
            prelogits, _ = inception_resnet_v1.inference(image_batch, self.keep_probability,
                phase_train=phase_train_placeholder, weight_decay=self.weight_decay)

            bottleneck = slim.fully_connected(prelogits, self.embedding_size, activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    weights_regularizer=slim.l2_regularizer(self.weight_decay),
                    normalizer_fn=slim.batch_norm,
                    normalizer_params_cnn=batch_norm_params_cnn,
                    scope='Bottleneck', reuse=False)  ### siz of embeddings
                    
            bottleneck = tf.concat([bottleneck, tf.cast(descriptors_batch, tf.float32)], 1)

            w_brand = facenet.weight_variable([bottleneck.get_shape().as_list()[1], nrof_brands], 'w_brand')
            b_brand = facenet.bias_variable([nrof_brands], 'b_brand')

            w_sku = facenet.weight_variable([bottleneck.get_shape().as_list()[1], nrof_classes], 'w_sku')
            b_sku = facenet.bias_variable([nrof_classes], 'b_sku')

            logits_brand = tf.matmul(bottleneck, w_brand) + b_brand
            logits_sku = tf.matmul(bottleneck, w_sku) + b_sku
            
            prediction_upc = tf.argmax(tf.nn.softmax(logits_sku), 1, name = "pred_upc")
            prediction_brand = tf.argmax(tf.nn.softmax(logits_brand), 1)

            accuracy_sku = tf.reduce_mean(tf.cast(tf.equal(prediction_upc, label_batch), tf.float32))
            accuracy_brand = tf.reduce_mean(tf.cast(tf.equal(prediction_brand, labels_brand_batch), tf.float32))

            tf.summary.scalar("accuracy brand", accuracy_brand)
            tf.summary.scalar("accuracy sku", accuracy_sku)
            
            embeddings = tf.nn.l2_normalize(bottleneck, 1, 1e-10, name='embeddings')

            # Add center loss
            if self.center_loss_factor> 0.0:
                prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, self.center_loss_alfa, nrof_classes)
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * self.center_loss_factor)

            if self.brand_loss_factor> 0.0:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= labels_brand_batch, logits=logits_brand, name='cross_entropy_per_example_brand')
                cross_entropy_mean_brand = tf.reduce_mean(cross_entropy, name='cross_entropy_brand')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, cross_entropy_mean_brand * self.brand_loss_factor)

            learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                self.learning_rate_decay_epochs*self.epoch_size, self.learning_rate_decay_factor, staircase=True)

            # Calculate the average cross entropy loss across the batch
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels= label_batch, logits=logits_sku, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)

            # Calculate the total losses
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

            # Build a Graph that trains the model with one batch of examples and updates the model parameters
            train_op = facenet.train(total_loss, global_step, self.optimizer,
                learning_rate, self.moving_average_decay, tf.global_variables(), True)

            # Create list with variables to restore
            restore_vars = []
            update_gradient_vars = []
            if self.pretrained_model:
                for var in tf.global_variables():
                    if not 'Embeddings/' in var.op.name and  not "Logits" in var.op.name and not "centers" in var.op.name :
                        restore_vars.append(var)
                    else:
                        update_gradient_vars.append(var)
            else:
                restore_vars = tf.global_variables()
                update_gradient_vars = tf.global_variables()

            restore_saver = tf.train.Saver(restore_vars)
            saver =  tf.train.Saver(tf.global_variables(), max_to_keep=3, write_version=tf.train.SaverDef.V1)

            # Build the summary operation based on the TF collection of Summaries.
            summary_op =  tf.summary.merge_all()

            # Start running operations on the Graph.
            config = tf.ConfigProto()
            config.log_device_placement=True
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            # Initialize variables
            sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
            sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

            summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
            tf.train.start_queue_runners(sess=sess)

            logging.error("-------   enter epoques  ------- ")
            with sess.as_default():

                if self.pretrained_model:
                    restore_saver.restore(sess, os.path.expanduser(self.pretrained_model))

                # Training and validation loop
                epoch = 0
                while epoch < self.max_nrof_epochs:
                    start_epoque = time.time()
                    step = sess.run(global_step, feed_dict=None)

                   # Train for one epoch
                    self.train(sess, epoch, self.im_list_train , self.liste_labels_upc_encoded_train, self.liste_labels_brand_encoded_train, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, labels_brand_placeholder,
                               learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, prediction_brand, prediction_upc,
                               total_loss, train_op, summary_op, summary_writer, regularization_losses, accuracy_sku, accuracy_brand, label_batch, labels_brand_batch)


                    # Save variables and the metagraph if it doesn't exist already
                    self.save_variables_and_metagraph(sess, saver, summary_writer, self.model_dir, step)

                    if epoch%2 == 0 or epoch == self.max_nrof_epochs-1:
                        self.evaluate(sess, embeddings, batch_size_placeholder, learning_rate_placeholder, prediction_brand, prediction_upc, label_batch, labels_brand_batch,
                                      phase_train_placeholder, enqueue_op, self.batch_size, step, summary_writer, epoch, saver, embedding_var, accuracy_sku, accuracy_brand)

                    print("time epoque : %s" %str(time.time() - start_epoque))
                    epoch +=1
                    
            sess.close()


    def train(self, sess, epoch, image_list, label_list, label_brand_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, labels_brand_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, prediction_brand, prediction_upc,
          loss, train_op, summary_op, summary_writer, regularization_losses, accuracy_sku, accuracy_brand, label_batch, labels_brand_batch):

        batch_number = 0
        index_epoch = sess.run(index_dequeue_op)

        label_epoch = np.array(label_list)[index_epoch]
        label_brand_epoch = np.array(label_brand_list)[index_epoch]
        image_epoch = np.array(image_list)[index_epoch]

        # Enqueue one epoch of image paths and labels
        labels_array = np.expand_dims(np.array(label_epoch), 1)
        labels_brand_array = np.expand_dims(np.array(label_brand_epoch), 1)
        image_paths_array = np.expand_dims(np.array(image_epoch), 1)
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, labels_brand_placeholder: labels_brand_array})

        # Training loop
        train_time = 0
        while batch_number < self.epoch_size:
            start_time = time.time()
            feed_dict = {learning_rate_placeholder: self.learning_rate, phase_train_placeholder:True, batch_size_placeholder: self.batch_size}
            if (batch_number % 100 == 0):
                err, _, step, reg_loss, summary_str, acc_sku, acc_brand, pred_brand, pred_upc, true_upc, true_brand  = sess.run([loss, train_op, global_step, regularization_losses, summary_op, accuracy_sku, accuracy_brand,  prediction_brand, prediction_upc, label_batch, labels_brand_batch], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)

            else:
                err, _, step, reg_loss, acc_sku, acc_brand = sess.run([loss, train_op, global_step, regularization_losses, accuracy_sku, accuracy_brand], feed_dict=feed_dict)
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f\t acc sku %2.3f\t acc brand %2.3f' %
                  (epoch, batch_number+1, self.epoch_size, duration, err, np.sum(reg_loss), acc_sku, acc_brand))
            batch_number += 1
            train_time += duration


    def evaluate(self, sess, embeddings, batch_size_placeholder, learning_rate_placeholder, prediction_brand, prediction_upc, label_batch, labels_brand_batch,
                 phase_train_placeholder, enqueue_op, batch_size, step, summary_writer, epoch, saver, embedding_var, accuracy_sku, accuracy_brand):

        X_train, Y_train = self.Create_comparaison_index(batch_size_placeholder, self.liste_labels_upc_encoded_train, self.images_train, self.descriptors_train, sess, embeddings, phase_train_placeholder, epoch, prediction_brand, prediction_upc, label_batch, labels_brand_batch)
        X_test, Y_test  = self.Create_comparaison_index(batch_size_placeholder, self.liste_labels_upc_encoded_test, self.images_test, self.descriptors_test, sess, embeddings, phase_train_placeholder, epoch, prediction_brand, prediction_upc, label_batch, labels_brand_batch)

        #### SGD classifier to have strong probabilities
        SGD = linear_model.SGDClassifier(loss='log', n_jobs=-1).fit(X_train, Y_train)
        proba = SGD.predict_proba(X_test)
        proba = [np.max(x) for x in proba]

        rep_sgd = pd.DataFrame({'pred_upc' : SGD.predict(X_test), 'True' : Y_test})
        logging.error('softmax scikit ------- > Accuracy upc: %1.3f ' %(sum(rep_sgd['pred_upc'] == rep_sgd['True'])/float(len(Y_test))))
        
        #### KNN 256 features X and classes Y
        neigh = KNeighborsClassifier(n_neighbors=1, n_jobs =-1)
        neigh.fit(X_train, Y_train)

        preds = neigh.predict(X_test)
        rep = pd.DataFrame({'pred' : preds , 'True' : Y_test , "proba": proba})
        rep["true_brand"] = pd.DataFrame(self.le.inverse_transform(Y_test))[0].apply(lambda x: x.split("/")[0])
        rep["pred_brand"] = pd.DataFrame(self.le.inverse_transform(preds))[0].apply(lambda x: x.split("/")[0])

        rep["rep_upc"] = 0
        rep.loc[rep['pred'] == rep['True'] ,"rep_upc"] = 1
        rep["rep_brand"] = 0
        rep.loc[rep["pred_brand"] == rep["true_brand"], "rep_brand"] = 1

        logging.error('KNN ------- > Accuracy upc: %1.3f Accuracy brand: %1.3f [%i/%i]' % (sum(rep["rep_upc"])/float(len(Y_test)), sum(rep["rep_brand"])/float(len(Y_test)), sum(rep["rep_upc"]), len(Y_test)))

        #### SVC 256 features X and classes Y
        clf = LinearSVC(C=1, multi_class='ovr').fit(X_train, Y_train)
        preds = clf.predict(X_test)

        svc_rep = pd.DataFrame({'pred' : preds, 'True' : Y_test, "proba": proba})
        svc_rep["rep_upc"] =0
        svc_rep.loc[svc_rep["pred"] == svc_rep["True"], "rep_upc"] =1
        logging.error("SVM       ------- > Accuracy upc: %1.3f"%(sum(svc_rep["pred"] == svc_rep["True"])/float(len(Y_test))))

        #### save models and parameters for future test
        if os.path.exists(self.model_dir + "/index_shelf.pkl"):
            os.remove(self.model_dir + "/index_shelf.pkl")

        if os.path.exists(self.model_dir + "/index_shelf_scv.pkl"):
            os.remove(self.model_dir + "/index_shelf_scv.pkl")

        if os.path.exists(self.model_dir + "/index_shelf_sgd.pkl"):
            os.remove(self.model_dir + "/index_shelf_sgd.pkl")

        joblib.dump(neigh, self.model_dir + "/index_shelf.pkl", compress=9)
        joblib.dump(clf, self.model_dir + "/index_shelf_svc.pkl", compress=9)
        joblib.dump(SGD, self.model_dir + "/index_shelf_sgd.pkl", compress=9)

        ### calculate treshold for 95% accuracy
        look_for_treshold = []
        for i in np.arange(1,0.3,-0.02):
            look_for_treshold.append([i, np.mean(svc_rep.loc[svc_rep["proba"]>i, "rep_upc"])])

        try:
            treshold = look_for_treshold[np.max(np.where(np.array(zip(*look_for_treshold)[1])> 0.95))][0]
            logging.error("treshold" + str(treshold))

        except Exception:
            treshold = 0.9
            pass

        ### model parameters
        data = {"embedding_size" :self.embedding_size,
                "image_size" : self.image_size,
                "network" : 'inception_resnet_v1',
                "learning_rate" : self.learning_rate,
                "keep_probability" : self.keep_probability,
                "treshold" : treshold,
                "center_loss_factor" : self.center_loss_factor,
                "center_loss_alfa" : self.center_loss_alfa,
                "brand_loss_factor" : self.brand_loss_factor
                }

        ##########################################   Summary  ###############################################
        brands = facenet.test_analysis(rep, epoch)

        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        summary.value.add(tag='Brand/Accuracy_global_brand_knn', simple_value= sum(rep["rep_brand"])/float(len(Y_test)))
        for index_brand in brands.index:
            summary.value.add(tag='Brand/Accuracy_brand_%s'%(index_brand), simple_value= brands.ix[index_brand].values[0])
        summary_writer.add_summary(summary, step)

        # Skus summary
        summary = tf.Summary()
        summary.value.add(tag='Sku/Accuracy KNN', simple_value= sum(rep["rep_upc"])/float(len(Y_test)))
        summary.value.add(tag='Sku/Accuracy SVC', simple_value= sum(svc_rep["pred"] == svc_rep["True"])/float(len(Y_test)))
        summary.value.add(tag='Sku/Accuracy SGD (logit)', simple_value= sum(rep_sgd['pred_upc'] == rep_sgd['True'])/float(len(Y_test)))
        summary_writer.add_summary(summary, step)

        #### write parameters json
        with open(self.model_dir + '/parameters.json', 'w') as outfile:
            json.dump(data, outfile)

        ##############  embeddings  ##############################
        sess.run(embedding_var, {embedding_var : X_test})
        summary_writer = tf.summary.FileWriter(self.model_dir)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        embedding.metadata_path = os.path.join(self.model_dir, 'metadata.tsv')
        metadata = os.path.join(self.model_dir, 'metadata.tsv')

        with open(metadata, 'w') as metadata_file:
            metadata_file.write('Id\tClasse\n')
            for i, row in enumerate(Y_test):
                metadata_file.write('%i\t%s\n' %(i,str(row)))

        embedding.sprite.image_path = os.path.join(self.model_dir, 'sprite.png')
        embedding.sprite.single_image_dim.extend([self.image_size, self.image_size])

        projector.visualize_embeddings(summary_writer, config)
        sprite = self.images_to_sprite(self.images_test)
        scipy.misc.imsave(os.path.join(self.model_dir, 'sprite.png'), sprite)


    def save_variables_and_metagraph(self, sess, saver, summary_writer, model_dir, step):

        start_time = time.time()
        checkpoint_path = os.path.join(model_dir, 'model-facenet.ckpt')
        saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
        print('Variables saved in %s seconds' %str(time.time() - start_time))

        if step >= self.max_nrof_epochs:
            saver.save(sess, checkpoint_path)


    def Create_comparaison_index(self, batch_size_placeholder, liste_labels, images, descriptors, sess, embeddings, phase_train_placeholder, epoch, prediction_brand, prediction_upc, label_batch, labels_brand_batch):

        print("length images : %i"%len(images))
        print("length labels : %i"%len(liste_labels))
        print("Number total classes is %i : "%len(self.le.classes_))

        # Run forward pass to calculate embeddings
        batch_size = min(600, len(liste_labels))
        nrof_images = len(liste_labels)
        nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embeddings.get_shape()[1]))

        for i in range(nrof_batches):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)
            batch_images= images[start_index:end_index, :, :, :]
            descriptors_i = np.array(descriptors)[start_index:end_index, :]
            emb_array[start_index:end_index,:] = sess.run(embeddings, {phase_train_placeholder : True,  "image_batch:0": batch_images, "descriptors_batch:0" : descriptors_i})

        return emb_array, liste_labels


    def load_data(self, image_paths):

        nrof_samples = len(image_paths)
        liste_paths =[]
        img_list = []
        descriptors = []
        for i in xrange(nrof_samples):
            try:
                img, description =self.image_descriptors([image_paths[i]], "test")
                descriptors.append(list(description))
                img_list.append(facenet.prewhiten(img))
                liste_paths.append(image_paths[i])
                if i%500 ==0:
                   print("image  %i/%i with shape %s"%(i, nrof_samples, img_list[i].shape))

            except Exception, e:
                print("could not open picture %s"%image_paths[i])
                print(e)
                pass

        images = np.stack(img_list)
        return images, descriptors, liste_paths


    def images_to_sprite(self, data):
        """Creates the sprite image along with any necessary padding

        Args:
          data: NxHxW[x3] tensor containing the images.

        Returns:
          data: Properly shaped HxWx3 image with any necessary padding.
        """
        if len(data.shape) == 3:
            data = np.tile(data[...,np.newaxis], (1,1,1,3))
        data = data.astype(np.float32)
        min = np.min(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
        max = np.max(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
        # Inverting the colors seems to look better for MNIST
        #data = 1 - data

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                constant_values=0)
        # Tile the individual thumbnails into an image.
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)
        return data


    def image_descriptors(self, filename, mode):
        
        image= cv2.imread(filename[0])/255.0
        
        ratio = image.shape[0]/float(image.shape[1])   

        means = [np.mean(image[:,:,0].ravel()), np.mean(image[:,:,1].ravel()), np.mean(image[:,:,2].ravel())]
        stds  = [np.std(image[:,:,0].ravel()), np.std(image[:,:,1].ravel()), np.std(image[:,:,2].ravel())]
         
        mins = [image[:,:,0].min(), image[:,:,1].min(), image[:,:,2].min()]
        maxs = [image[:,:,0].max(), image[:,:,1].max(), image[:,:,2].max()]
        
        kurt = [scipy.stats.kurtosis(image[:,:,0].ravel()), scipy.stats.kurtosis(image[:,:,1].ravel()), scipy.stats.kurtosis(image[:,:,2].ravel())]
        skew = [scipy.stats.skew(image[:,:,0].ravel()), scipy.stats.skew(image[:,:,1].ravel()), scipy.stats.skew(image[:,:,2].ravel())]
        
        out = means + stds + mins + maxs + kurt + skew + [ratio]

        if mode == "train":
            return np.array(out)*0.5
            
        if mode == "test":
            image = cv2.resize(image, (self.image_size, self.image_size))
            return image, np.array(out)*0.5
