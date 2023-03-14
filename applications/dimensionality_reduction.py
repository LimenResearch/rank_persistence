import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd
import sys
import numpy as np

class EmbeddingVisualiser(object):
    def __init__(self, labels, features, label_names = None,
                 feature_names = None, session_name = ""):
        self.session_name = session_name
        self.labels = labels if isinstance(labels, list) else [labels]
        self.label_names = label_names if isinstance(label_names, list) else [label_names]

        if isinstance(features, list):
            self.embeddings = []
            if feature_names is None:
                feature_names = [str(i) for i in range(len(features))]

            for f, name in zip(features, feature_names):
                self.embeddings.append(tf.Variable(f, name = name))
        else:
            if feature_names is None:
                name = "0"
            elif isinstance(feature_names, list):
                name = feature_names[0]
            else:
                name = feature_names

            self.embeddings = [tf.Variable(features, name = name)]


    def create_labels_file(self, embeddings_folder):
        embeddings_folder = os.path.join(embeddings_folder,
                                         'checkpoints_' + self.session_name)
        if not os.path.isdir(embeddings_folder):
            os.makedirs(embeddings_folder)
        path = os.path.join(embeddings_folder, 'labels.csv')

        for labels, name in zip(self.labels, self.label_names):

            df = pd.DataFrame(labels)
            df.to_csv(path, sep='\t', header=self.label_names)


    def create_sprite_file(self, images, labels, name):
        """
        Generates sprite image and associated labels
        """
        number_of_rows = len([l for l in labels if l==0])
        number_of_columns = len(np.unique(labels))
        number_of_images = int(number_of_columns * number_of_rows)
        im_height, im_width = images.shape[1:3]
        sprite_height, sprite_width = im_height * number_of_rows, im_width * number_of_columns

        row_sprite = []
        sprite = []
        i = 0

        while i < number_of_images:
            row_sprite.append(images[i])
            if (i+1) % number_of_columns == 0:
                sprite.append(np.hstack(row_sprite))
                row_sprite = []
            i += 1

        sprite = np.vstack(sprite)
        sprite = (sprite * 255).astype(np.uint8)
        cv2.imwrite(name + '.png', sprite)
        return images, labels


    def visualize(self, embeddings_folder):
        step = tf.Variable(0, name='step', trainable=False)
        saver = tf.train.Saver()
        checkpoint_folder = os.path.join(embeddings_folder,
                                         'checkpoints_' + self.session_name)
        if os.path.isdir(checkpoint_folder) == False:
            os.makedirs(checkpoint_folder)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            # Use the same checkpoint_folder where you stored your checkpoint.
            summary_writer = tf.compat.v1.summary.FileWriter(checkpoint_folder)
            # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
            config = projector.ProjectorConfig()
            for e in self.embeddings:
                f = config.embeddings.add()
                f.tensor_name = e.name
                f.metadata_path = 'labels.csv'
            # You can add multiple embeddings. Here we add only one.
            projector.visualize_embeddings(summary_writer, config)
            step.assign(0).eval()
            saver.save(session, os.path.join(checkpoint_folder, "model.ckpt"), step)

