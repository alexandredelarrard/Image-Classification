# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:56:12 2017

@author: alexandre
"""

import os.path
import time
import tensorflow as tf
import numpy as np

import logging
from tensorflow.python.ops import data_flow_ops
import math
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from facenet_utils.models import inception_resnet_v1
from facenet_utils import facenet


class Train_price_facenet(object):

    def __init__(self, global_parameters, pictures_dict, params_cnn):
        
        im_list_path = pictures_dict["pictures_train_facenet"]        
        
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
        
        self.path_client        =  "/".join([global_parameters["ibm_client_train_data_path_image_reco"], global_parameters["fam_id"]]) + "/"
        self.model_dir          =  global_parameters['path_to_model_facenet']
        self.summary_dir        =  os.path.join("/".join([global_parameters['path_to_model_facenet'], "tmp", "summary_dashboard"]), time.strftime("%H:%M:%S"))
        self.global_parameters  =  global_parameters

        self.Main(im_list_path)


    def Flow_tensorflow(self, im_list_path):

         image_path_train, image_path_test, length_labels_train, length_labels_test, tags_train, tags_test = self.shape_dataset(im_list_path, self.proportion_train_test)
    
         print(" Train number pictures : %i"%len(image_path_train))
         print(" Test number pictures  : %i"%len(image_path_test))
         num_labels = 11
    
         with tf.Graph().as_default():
            tf.set_random_seed(self.seed)
            global_step = tf.Variable(0, trainable=False)
    
            labels = ops.convert_to_tensor(length_labels_train, dtype=tf.int32)
            range_size = array_ops.shape(labels)[0]
            index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=self.seed, capacity=32)
    
            index_dequeue_op = index_queue.dequeue_many(self.batch_size*self.epoch_size, 'index_dequeue')
    
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
    
            ### feed variables placeholder
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            label_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='length')
            coordinates_placeholder = tf.placeholder(tf.int32, shape=(None, 6), name='coordinates')
    
            input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                            dtypes=[tf.string, tf.int32, tf.int32],
                                            shapes=[(1,), (1,), (6,)],
                                            shared_name=None, name=None)
    
            enqueue_op = input_queue.enqueue_many([image_paths_placeholder, label_placeholder, coordinates_placeholder], name='enqueue_op')
    
            nrof_preprocess_threads = 4
            images_and_labels = []
            for _ in range(nrof_preprocess_threads):
                filenames, label, coordinates = input_queue.dequeue()
                images = []
                for filename in tf.unstack(filenames):
    
                    file_contents = tf.read_file(filename)
                    image = tf.image.decode_jpeg(file_contents, fancy_upscaling=True)
                    image = tf.image.resize_images(image, [int(self.image_size*1.1), int(self.image_size*1.1)])
    
                    image = tf.random_crop(image, [self.image_size, self.image_size, 3])
                    image = tf.py_func(facenet.random_rotate_image, [image, 10.0], tf.uint8)
    
                    if np.random.randint(0,10)>= 5:
                        image = tf.image.random_brightness(image, max_delta=32)
    
                    if np.random.randint(0,10)>= 5:
                        image = tf.image.random_contrast(image, lower=0.5, upper=1.)
    
                    if np.random.randint(0,10)>= 5:
                        image = tf.image.random_saturation(image, 0.5, 1.)
    
                    if np.random.randint(0,10)>= 5:
                        image = tf.image.random_hue(image, 0.3)
    
                    image.set_shape((self.image_size, self.image_size, 3))
                    images.append(tf.image.per_image_standardization(image))
    
                images_and_labels.append([images, label, [coordinates]])
    
            image_batch, labels_batch, labels_number_batch = tf.train.batch_join(
                images_and_labels, batch_size=batch_size_placeholder,
                shapes=[(self.image_size, self.image_size, 3), (), (6,)], enqueue_many=True,
                capacity= 4 * nrof_preprocess_threads * self.batch_size,
                allow_smaller_final_batch=True)
    
            image_batch = tf.identity(image_batch, 'image_batch')
            labels_batch =  tf.identity(labels_batch, 'length_batch')
            labels_number_batch = tf.identity(labels_number_batch, 'coord_batch')
    
            # Build the inference graph -- >1792 embeddings = H
            prelogits, end_points = inception_resnet_v1.inference(image_batch, self.keep_probability,
                    phase_train=phase_train_placeholder, weight_decay=self.weight_decay)
    
            learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                    self.learning_rate_decay_epochs*self.epoch_size, self.learning_rate_decay_factor, staircase=True)
    
            bottleneck = end_points['prelogits'].get_shape().as_list()[1]
    
            # Softmax length
            w_fc = facenet.weight_variable([bottleneck, self.nbr_class_length], 'w_fc')
            b_fc = facenet.bias_variable([self.nbr_class_length], 'b_fc')
    
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
    
            with tf.name_scope("softmax_length"):
                logits_length = tf.matmul(prelogits, w_fc) + b_fc
            with tf.name_scope("softmax_1"):
                logits_1 = tf.matmul(prelogits, w_s1) + b_s1
            with tf.name_scope("softmax_2"):
                logits_2 = tf.matmul(prelogits, w_s2) + b_s2
            with tf.name_scope("softmax_3"):
                logits_3 = tf.matmul(prelogits, w_s3) + b_s3
            with tf.name_scope("softmax_4"):
                logits_4 = tf.matmul(prelogits, w_s4) + b_s4
            with tf.name_scope("softmax_5"):
                logits_5 = tf.matmul(prelogits, w_s5) + b_s5
            with tf.name_scope("softmax_5"):
                logits_6 = tf.matmul(prelogits, w_s6) + b_s6
    
            accuracy_length = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits_length, 1), tf.int32), tf.cast(labels_batch, tf.int32)), tf.float32))
            accuracy_first_digit = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits_1, 1), tf.int32), tf.cast(labels_number_batch[:, 0], tf.int32)), tf.float32))
            accuracy_second_digit = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits_2, 1), tf.int32), tf.cast(labels_number_batch[:, 1], tf.int32)), tf.float32))
            accuracy_third_digit = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits_3, 1), tf.int32), tf.cast(labels_number_batch[:, 2], tf.int32)), tf.float32))
            accuracy_fourth_digit = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits_4, 1), tf.int32), tf.cast(labels_number_batch[:, 3], tf.int32)), tf.float32))
            accuracy_fifth_digit = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits_5, 1), tf.int32), tf.cast(labels_number_batch[:, 4], tf.int32)), tf.float32))
            accuracy_sixth_digit = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(logits_6, 1), tf.int32), tf.cast(labels_number_batch[:, 5], tf.int32)), tf.float32))
    
            prediction = tf.stack([tf.cast(tf.argmax(logits_1, 1), tf.int32), tf.cast(tf.argmax(logits_2, 1), tf.int32), tf.cast(tf.argmax(logits_3, 1), tf.int32),
                                  tf.cast(tf.argmax(logits_4, 1), tf.int32), tf.cast(tf.argmax(logits_5, 1), tf.int32), tf.cast(tf.argmax(logits_6, 1), tf.int32)], axis=1)
    
            true_value = tf.stack([labels_number_batch[:, 0], labels_number_batch[:, 1], labels_number_batch[:, 2],
                                  labels_number_batch[:, 3], labels_number_batch[:, 4], labels_number_batch[:, 5]], axis=1)
    
            total_accuracy  = tf.reduce_mean(tf.cast(tf.equal(tf.cast(prediction, tf.int32), tf.cast(true_value, tf.int32)), tf.float32))
            accuracy = [total_accuracy, accuracy_length, accuracy_first_digit, accuracy_second_digit, accuracy_third_digit, accuracy_fourth_digit, accuracy_fifth_digit, accuracy_sixth_digit]
    
            with tf.name_scope("loss"):
                loss_length = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_length, labels=labels_batch)
                loss_soft_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=labels_number_batch[:, 0])
                loss_soft_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=labels_number_batch[:, 1])
                loss_soft_3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=labels_number_batch[:, 2])
                loss_soft_4 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_4, labels=labels_number_batch[:, 3])
                loss_soft_5 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_5, labels=labels_number_batch[:, 4])
                loss_soft_6 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_6, labels=labels_number_batch[:, 5])
    
                loss_soft_1 = tf.where(tf.is_nan(loss_soft_1), tf.zeros_like(loss_soft_1), loss_soft_1)
                loss_soft_2 = tf.where(tf.is_nan(loss_soft_2), tf.zeros_like(loss_soft_2), loss_soft_2)
                loss_soft_3 = tf.where(tf.is_nan(loss_soft_3), tf.zeros_like(loss_soft_3), loss_soft_3)
                loss_soft_4 = tf.where(tf.is_nan(loss_soft_4), tf.zeros_like(loss_soft_4), loss_soft_4)
                loss_soft_5 = tf.where(tf.is_nan(loss_soft_5), tf.zeros_like(loss_soft_5), loss_soft_5)
                loss_soft_6 = tf.where(tf.is_nan(loss_soft_6), tf.zeros_like(loss_soft_6), loss_soft_6)
    
                loss = tf.reduce_mean(loss_length + loss_soft_1 + loss_soft_2 + loss_soft_3 + loss_soft_4 + loss_soft_5 + loss_soft_6)
    
            losses = [loss_length, loss_soft_1, loss_soft_2, loss_soft_3, loss_soft_4, loss_soft_5, loss_soft_6]
    
            train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0).minimize(
                                loss, global_step=global_step)
    
            saver =  tf.train.Saver(tf.global_variables(), max_to_keep=3, write_version=tf.train.SaverDef.V1)
            summary_op =  tf.summary.merge_all()
    
            config = tf.ConfigProto()
            config.log_device_placement=True
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
    
            # Initialize variables
            sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
            sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})
    
            summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
            tf.train.start_queue_runners(sess=sess)
    
            print("enter epoques")
            with sess.as_default():
                epoch = 0
                while epoch < self.num_epoch:
                    start_epoque = time.time()
                    step = sess.run(global_step, feed_dict=None)
    
                    # Train for one epoch
                    self.train(sess, epoch, image_path_train, length_labels_train, tags_train,
                          index_dequeue_op, enqueue_op, image_paths_placeholder, label_placeholder, coordinates_placeholder,
                          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                          loss, train_op, summary_op, summary_writer, accuracy, losses)
    
                    # Save variables and the metagraph if it doesn't exist already
                    self.save_variables_and_metagraph(sess, saver, summary_writer, self.summary_dir, step)
    
                    self.evaluate(sess, epoch, image_path_test, length_labels_test, tags_test,
                          index_dequeue_op, enqueue_op, image_paths_placeholder, label_placeholder, coordinates_placeholder,
                          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                          loss, train_op, summary_op, summary_writer, accuracy, losses, step, prediction, true_value)
    
                    print("time epoque : %s" %str(time.time() - start_epoque))
                    epoch +=1
    
                sess.close()


    def train(self, sess, epoch, image_path_train, length_labels_train, tags_train,
          index_dequeue_op, enqueue_op, image_paths_placeholder, label_placeholder, coordinates_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, accuracy, losses):

        batch_number = 0
        index_epoch = sess.run(index_dequeue_op)

        label_epoch = np.array(length_labels_train)[index_epoch]
        label_coord_epoch = np.array(tags_train)[index_epoch]
        image_epoch = np.array(image_path_train)[index_epoch]

        # Enqueue one epoch of image paths and labels
        labels_array = np.expand_dims(np.array(label_epoch), 1)
        image_paths_array = np.expand_dims(np.array(image_epoch), 1)
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, label_placeholder: labels_array, coordinates_placeholder: label_coord_epoch})

        # Training loop
        train_time = 0
        while batch_number < self.epoch_size:
            start_time = time.time()
            feed_dict = {learning_rate_placeholder: self.learning_rate, phase_train_placeholder:True, batch_size_placeholder: self.batch_size}

            err, _, step, summary_str, total_accuracy, acc_length, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6 = sess.run([loss, train_op, global_step, summary_op] + accuracy, feed_dict=feed_dict)

            duration = time.time() - start_time
            print('[%d][%d/%d]\tT %.3f\tLoss %2.3f\tAcc L. %2.3f\tSoft1 %2.3f\tSoft2 %2.3f\tSoft3 %2.3f\tSoft4 %2.3f\tTotal acc:  %2.3f' %
                  (epoch, batch_number+1, self.epoch_size, duration, err, acc_length, acc_1, acc_2, acc_3, acc_4, total_accuracy))

            batch_number += 1
            train_time += duration

        return step


    def evaluate(sess, epoch, image_path_test, length_labels_test, tags_test,
          index_dequeue_op, enqueue_op, image_paths_placeholder, label_placeholder, coordinates_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, accuracy, losses, step, prediction, true_value):

        image_test = Image_from_path(image_path_test, self.image_size)

        label_epoch = np.array(length_labels_test)
        label_coord_epoch = np.array(tags_test)

        # Training loop
        batch_size = 600
        nrof_batch = int(math.ceil(1.0*len(image_path_test) / batch_size))
        acc_length_tot = 0
        acc_1_tot = 0
        acc_2_tot = 0
        acc_3_tot = 0
        acc_4_tot = 0
        acc_5_tot = 0
        acc_6_tot = 0
        total_accuracy_tot = 0
        for i in range(nrof_batch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, len(image_path_test))
            feed_dict = {learning_rate_placeholder: 1., phase_train_placeholder:False, "image_batch:0": image_test[start_index:end_index, :, :, :], "length_batch:0": label_epoch[start_index:end_index], "coord_batch:0": label_coord_epoch[start_index:end_index,:]}
            total_accuracy, acc_length, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, pred, true = sess.run(accuracy + [prediction] + [true_value], feed_dict=feed_dict)
            acc_1_tot += acc_1
            acc_2_tot += acc_2
            acc_3_tot += acc_3
            acc_4_tot += acc_4
            acc_5_tot += acc_5
            acc_6_tot += acc_6
            acc_length_tot += acc_length
            total_accuracy_tot += total_accuracy

        print(" ----------------   Test  ------------------")
        print('Acc length %2.3f\tSoft1 %2.3f\tSoft2 %2.3f\tSoft3 %2.3f\tSoft4 %2.3f\tTotal acc:  %2.3f' %
              (acc_length_tot/nrof_batch, acc_1_tot/nrof_batch, acc_2_tot/nrof_batch, acc_3_tot/nrof_batch, acc_4_tot/nrof_batch, total_accuracy/nrof_batch))

        print(" True base ground ")
        print(len(true))
        print(true[0:20])
        print(" Prediction base ground ")
        print(len(pred))
        print(pred[0:20])

        summary = tf.Summary()
        summary.value.add(tag='Test Accuracy/Length', simple_value= acc_length_tot/nrof_batch)
        summary.value.add(tag='Test Accuracy/Softmax 1', simple_value= acc_1_tot/nrof_batch)
        summary.value.add(tag='Test Accuracy/Softmax 2', simple_value= acc_2_tot/nrof_batch)
        summary.value.add(tag='Test Accuracy/Softmax 3', simple_value= acc_3_tot/nrof_batch)
        summary.value.add(tag='Test Accuracy/Softmax 4', simple_value= acc_4_tot/nrof_batch)
        summary.value.add(tag='Test Accuracy/Softmax 5', simple_value= acc_5_tot/nrof_batch)
        summary.value.add(tag='Test Accuracy/Softmax 6', simple_value= acc_6_tot/nrof_batch)
        summary_writer.add_summary(summary, step)


    def save_variables_and_metagraph(self, sess, saver, summary_writer, model_dir, step):

        start_time = time.time()
        checkpoint_path = os.path.join(model_dir, 'model-facenet.ckpt')
        saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
        print('Variables saved in %s seconds' %str(time.time() - start_time))

        if step >= self.max_nrof_epochs:
            saver.save(sess, checkpoint_path)


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
