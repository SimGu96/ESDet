from tensorflow.keras.layers import Conv2D, BatchNormalization, concatenate, MaxPool2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.regularizers import l2
from .blocks import DownsamplingBlock, FCU, PFCU, UpsamplingBlock



class ESDet():

    def __init__(self, config, input_shape, output_shape):
        """Init of SqueezeDet Class
        
        Arguments:
            config {[type]} -- dict containing hyperparameters for network building
        """

        #hyperparameter config file
        self.config = config

        #create Keras model
        self.model = self._create_model(input_shape, output_shape)


    def get_head(self, backbone, output_shape):
        x = BatchNormalization()(backbone)
        x = self.SqueezeDet_fire_layer("fire1", x, 64, 128, 128)
        x = BatchNormalization()(x)
        x = self.SqueezeDet_fire_layer("fire2", x, 64, 128, 128)
        x = BatchNormalization()(x)
        x = self.SqueezeDet_fire_layer("fire3", x, 64, 128, 128)

        # Add pooling to get N_ANCHORS_WIDTH x N_ANCHORS_HEIGHT output size.
        # Maybe better to add stride in Conv2D instead?
        x = MaxPool2D(pool_size=(4, 4), strides=(4, 4), name="pool_head")(x)  


        head = Conv2D(
                name='OD_head', filters=output_shape, kernel_size=(3, 3), 
                strides=(1, 1), activation=None, padding="SAME", use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.001),
            kernel_regularizer=l2(self.config.WEIGHT_DECAY))(x)


        head = tf.keras.layers.Reshape((self.config.ANCHORS, -1))(head)

        return head

    
    def get_backbone(self, input_shape):
        inputs = tf.keras.layers.Input(input_shape)
        ##### Encoder #####
        # Block 1
        x = DownsamplingBlock(inputs, 3, 16)
        x = FCU(x, 16, K=3)
        x = FCU(x, 16, K=3)
        x = FCU(x, 16, K=3)
        # Block 2
        x = DownsamplingBlock(x, 16, 64)
        x = FCU(x, 64, K=5)
        x = FCU(x, 64, K=5)
        # Block 3
        x = DownsamplingBlock(x, 64, 128)
        x = PFCU(x, 128)
        x = PFCU(x, 128)
        x = PFCU(x, 128)
        ##### Decoder #####
        # Block 4
        x = UpsamplingBlock(x, 128)
        x = FCU(x, 128, K=5, dropout_prob=0.0)
        x = FCU(x, 128, K=5, dropout_prob=0.0)
        # Block 5
        #x = UpsamplingBlock(x, 16)
        #x = FCU(x, 16, K=3, dropout_prob=0.0)
        #x = FCU(x, 16, K=3, dropout_prob=0.0)
        #output = tf.keras.layers.Conv2DTranspose(
        #    output_channels, 3, padding='same',
        #    strides=(2, 2), use_bias=True
        #)(x)
        #x = FCU(output, output_channels, K=3, dropout_prob=0.0)

        return inputs, x



    def _create_model(self, input_shape, output_shape):

        inputs, backbone = self.get_backbone(input_shape)
        head = self.get_head(backbone, output_shape)
        model = Model(inputs=inputs, outputs=head)

        return model


    def SqueezeDet_fire_layer(self, name, input, s1x1, e1x1, e3x3):
        """
        wrapper for fire layer constructions

        :param name: name for layer
        :param input: previous layer
        :param s1x1: number of filters for squeezing
        :param e1x1: number of filter for expand 1x1
        :param e3x3: number of filter for expand 3x3
        :param stdd: standard deviation used for intialization
        :return: a keras fire layer
        """

        sq1x1 = Conv2D(
            name = name + '/squeeze1x1', filters=s1x1, kernel_size=(1, 1), strides=(1, 1),
            padding='SAME', activation="relu")(input)

        ex1x1 = Conv2D(
            name = name + '/expand1x1', filters=e1x1, kernel_size=(1, 1), strides=(1, 1),
            padding='SAME', activation="relu")(sq1x1)

        ex3x3 = Conv2D(
            name = name + '/expand3x3',  filters=e3x3, kernel_size=(3, 3), strides=(1, 1),
            padding='SAME', activation="relu")(sq1x1)

        return concatenate([ex1x1, ex3x3], axis=3)
