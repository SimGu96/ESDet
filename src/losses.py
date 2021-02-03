import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import src.utils_squeezeDet as utils


class ESDetBoxLoss(tf.losses.Loss):
    """Implements bounding box loss"""

    def __init__(self, config):
        super(ESDetBoxLoss, self).__init__(
            reduction="none", name="box_loss"
        )
        self._conf = config

    def call(self, y_true, y_pred):
        #slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_delta_input = y_true[:, :, 5:9]

        #number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        #before computing the losses we need to slice the network outputs
        _, _, pred_box_delta = utils.slice_predictions(y_pred, self._conf)

        #bounding box loss
        bbox_loss = (K.sum(self._conf.LOSS_COEF_BBOX * K.square(input_mask * (pred_box_delta - box_delta_input))) / num_objects)

        return bbox_loss


class ESDetClassLoss(tf.losses.Loss):
    """Implements classification loss"""

    def __init__(self, config):
        super(ESDetClassLoss, self).__init__(
            reduction="none", name="class_loss"
        )
        self._conf = config

    def call(self, y_true, y_pred):
        #slice y_true
        input_mask = y_true[:, :, 0]
        #input_mask = K.expand_dims(input_mask, axis=-1)
        labels = y_true[:, :, 9:]

        #number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        #before computing the losses we need to slice the network outputs
        pred_class_probs, _, _ = utils.slice_predictions(y_pred, self._conf)

        #compute class loss,add a small value into log to prevent blowing up
        class_loss_orig = K.sum(labels * (-K.log(pred_class_probs + self._conf.EPSILON))
                    + (1 - labels) * (-K.log(1 - pred_class_probs + self._conf.EPSILON))
        * input_mask * self._conf.LOSS_COEF_CLASS) / num_objects

        class_loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(labels, pred_class_probs) * input_mask * self._conf.LOSS_COEF_CLASS
        class_loss = K.sum(class_loss) / num_objects

        #tf.print(class_loss, class_loss_orig, num_objects, output_stream='file://loggtest.txt')

        return class_loss


class ESDetClassLossOrig(tf.losses.Loss):
    """Implements classification loss"""

    def __init__(self, config):
        super(ESDetClassLoss, self).__init__(
            reduction="none", name="class_loss_orig"
        )
        self._conf = config

    def call(self, y_true, y_pred):
        #slice y_true
        input_mask = y_true[:, :, 0]
        #input_mask = K.expand_dims(input_mask, axis=-1)
        labels = y_true[:, :, 9:]

        #number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        #before computing the losses we need to slice the network outputs
        pred_class_probs, _, _ = utils.slice_predictions(y_pred, self._conf)

        #compute class loss,add a small value into log to prevent blowing up
        class_loss_orig = K.sum(labels * (-K.log(pred_class_probs + self._conf.EPSILON))
                    + (1 - labels) * (-K.log(1 - pred_class_probs + self._conf.EPSILON))
        * input_mask * self._conf.LOSS_COEF_CLASS) / num_objects

        #tf.print(class_loss, class_loss_orig, num_objects, output_stream='file://loggtest.txt')

        return class_loss



class ESDetConfLoss(tf.losses.Loss):
    """Implements confidence loss"""

    def __init__(self, config):
        super(ESDetConfLoss, self).__init__(
            reduction="none", name="conf_Loss"
        )
        self._conf = config

    def call(self, y_true, y_pred):
        #slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]

        #number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        #before computing the losses we need to slice the network outputs
        _, pred_conf, pred_box_delta = utils.slice_predictions(y_pred, self._conf)

        #compute boxes
        det_boxes = utils.boxes_from_deltas(pred_box_delta, self._conf)

        #unstack is not avaible in pure keras backend
        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])

        #compute the ious
        ious = utils.tensor_iou(
            utils.bbox_transform(unstacked_boxes_pred),
            utils.bbox_transform(unstacked_boxes_input),
            input_mask,
            self._conf
        )

        #reshape input for correct broadcasting
        input_mask = K.reshape(input_mask, [self._conf.BATCH_SIZE, self._conf.ANCHORS])

        #confidence score loss
        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * self._conf.LOSS_COEF_CONF_POS / num_objects
                    + (1 - input_mask) * self._conf.LOSS_COEF_CONF_NEG / (self._conf.ANCHORS - num_objects)),
                axis=[1]
            ),
        )
        return conf_loss



class ESDetLoss(tf.losses.Loss):
    """Implements ESDet Loss"""

    def __init__(self, config):
        super(ESDetLoss, self).__init__(
            reduction="none", name="ESDetLoss"
        )
        self._conf = config
        self._boxLoss = ESDetBoxLoss(config)
        self._classLoss = ESDetClassLoss(config)
        self._confLoss = ESDetConfLoss(config)

    def call(self, y_true, y_pred):

        boxLoss = self._boxLoss(y_true, y_pred)
        classLoss = self._classLoss(y_true, y_pred)
        confLoss = self._confLoss(y_true, y_pred)

        return boxLoss + classLoss + confLoss
