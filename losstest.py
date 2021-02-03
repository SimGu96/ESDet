import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

y_true = tf.cast([[0, 1, 0], [0, 0, 1]], tf.float32)
y_pred = tf.cast([[0.05, 0.95, 0], [0.1, 0.8, 0.1]], tf.float32)
y_pred2 = np.clip(y_pred, 0.0000001, 1-0.0000001)
print(y_pred2)

cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_pred, y_true).numpy())

cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)
print(cce(y_pred, y_true).numpy())

cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE) #* [[1], [0]]
print(cce(y_pred, y_true).numpy())



class_loss = K.sum(y_true * (-K.log(y_pred2)) + (1.0 - y_true) * (-K.log(1.0 - y_pred2)))


print(class_loss.numpy())
print((y_true * (-K.log(y_pred2)) + (1.0 - y_true) * (-K.log(1.0 - y_pred2))).numpy())
#print(1.0 - y_true)
#print(K.log(1.0 - y_pred2).numpy())
#print(type(K.log(y_pred2).numpy()))


class_loss2 = K.sum(y_true * (-K.log(y_pred2))
                    + (1 - y_true) * (-K.log(1 - y_pred2))
        )

print(class_loss2)
