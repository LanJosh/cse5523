"""Unet model definition

This work was first described in:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Phillip Fischer, and Thomas Brox
"""

import tensorflow as tf

class Unet():
    
    def __init__(self):
        """Constructs the graph definition for the U-Net model."""

        ### Define the network.
        # Input are 572x572 RGB images.
        self.input = tf.placeholder(tf.float32, [None, 572, 572, 3])

        # Placeholder for the ground truth mask.
        self.y_ = tf.placeholder(tf.float32, [None, 388, 338, 2])

        # Down layers
        self.conv1 = tf.layers.conv2d(self.input, 64, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.272))
        self.conv1 = tf.layers.conv2d(self.conv1, 64, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0589))
        self.down1 = tf.layers.max_pool2d(self.conv1, 2, 2)

        self.conv2 = tf.layers.conv2d(self.down1, 128, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0589))
        self.conv2 = tf.layers.conv2d(self.conv2, 128, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0416))
        self.down2 = tf.layers.max_pool2d(self.conv2, 2, 2)

        self.conv3 = tf.layers.conv2d(self.down2, 256, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0416))
        self.conv3 = tf.layers.conv2d(self.conv3, 256, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0294))
        self.down3 = tf.layers.max_pool2d(self.conv3, 2, 2)

        self.conv4 = tf.layers.conv2d(self.down3, 512, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0294))
        self.conv4 = tf.layers.conv2d(self.conv4, 512, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0208))
        self.down4 = tf.layers.max_pool2d(self.conv4, 2, 2)

        self.conv5 = tf.layers.conv2d(self.down3, 1024, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0208))
        self.conv5 = tf.layers.conv2d(self.conv5, 1024, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0147))
        
        # Up layers
        self.conv6 = tf.layers.conv2d_transpose(self.conv5, 512, 2,
            activation=tf.nn.relu, 
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0221))
        self.crop1 = tf.slice(self.conv4, [0, 4, 4, 0], [-1, 56, 56, -1])
        self.copy1 = tf.concat(self.crop1, self.conv6, 3)
        self.conv6 = tf.layers.conv2d(self.copy1, 512, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0147))
        self.conv6 = tf.layers.conv2d(self.conv6, 512, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0208))

        self.conv7=tf.layers.conv2d_transpose(self.conv6, 256, 2,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0313))
        self.crop2 = tf.slice(self.conv3, [0, 16, 16, 0], [-1, 104, 104, -1])
        self.copy2 = tf.concat(self.crop2, self.conv7, 3)
        self.conv7 = tf.layers.conv2d(self.copy2, 256, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0208))
        self.conv7 = tf.layers.conv2d(self.conv7, 256, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0294))

        self.conv8 = tf.layers.conv2d_transpose(self.conv7, 128, 2,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0442))
        self.crop3 = tf.slice(self.conv2, [0, 40, 40, 0], [-1, 200, 200, -1])
        self.copy3 = tf.concat(self.crop3, self.conv8, 3)
        self.conv8 = tf.layers.conv2d(self.copy3, 128, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0294))
        self.conv8 = tf.layers.conv2d(self.conv8, 128, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0416))

        self.conv9 = tf.layers.conv2d_transpose(self.conv7, 64, 2,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0625))
        self.crop4 = tf.slice(self.conv2, [0, 88, 88, 0], [-1, 392, 392, -1])
        self.copy4 = tf.concat(self.crop3, self.conv8, 3)
        self.conv9 = tf.layers.conv2d(self.copy4, 64, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0416))
        self.conv9 = tf.layers.conv2d(self.conv9, 64, 3, activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=.0589))

        self.output = tf.layers.conv2d(self.conv9, 2, 1, activation=tf.nn.relu,
            kernel_initializer=tf.nn.sigmoid)

        # Define loss and train step
        self.loss = tf.divide(tf.multiply(
                tf.reduce_sum(tf.multiply(self.output, self.y_)), 2), 602176)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    def mask(self, sess, x):
        """Creates a segmentation mask for the image.

        Args:
          sess: A Tensorflow session object.
          x: An matrix of shape (1, N, N, 3)

        Returns:
          An Image object.
        """
        q = sess.run(self.output, {self.input: x})
        q = q[0,:,:,:]
        
        car_scores = q[:,:,0]
        bg_scores = q[:,:,1]
        mask = np.empty((338, 338,2))
        for i in range(338):
            for j in range(338):
                if car_scores[i][j] > bg_scores[i][j]:
                    mask[i][j][0] = 255 
                    mask[i][j][1] = 0
                else:
                    mask[i][j][0] = 0
                    mask[i][j][1] = 255 
        mask = Image.fromarray(mask)
        return mask 

