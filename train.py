import tensorflow as tf
from src.dataGenerator import generator_from_data_path
from Networks.ESDet import ESDet
from src.create_config import load_dict
from tensorflow.keras import backend as K
from src.losses import *


img_file = "img_train.txt"
gt_file = "gt_train.txt"
img_file_val = "img_val.txt"
gt_file_val = "gt_val.txt"
EPOCHS = 5
STEPS = None
REDUCELRONPLATEAU = True
OPTIMIZER = "adam"


#open files with images and ground truths files with full path names
with open(img_file) as imgs:
    img_names = imgs.read().splitlines()
imgs.close()
with open(gt_file) as gts:
    gt_names = gts.read().splitlines()
gts.close()

#open validation files with images and ground truths files with full path names
with open(img_file_val) as imgs:
    img_names_val = imgs.read().splitlines()
imgs.close()
with open(gt_file_val) as gts:
    gt_names_val = gts.read().splitlines()
gts.close()

config_file= "squeeze.config"
cfg = load_dict(config_file)
cfg.EPOCHS = EPOCHS
#cfg.BATCH_SIZE = 2



nbatches_train, mod = divmod(len(img_names), cfg.BATCH_SIZE)
nbatches_val, mod = divmod(len(img_names_val), cfg.BATCH_SIZE)



if STEPS is not None:
    nbatches_train = STEPS

cfg.STEPS = nbatches_train


#print some run info
print("Number of images: {}".format(len(img_names)))
print("Number of epochs: {}".format(EPOCHS))
print("Number of batches: {}".format(nbatches_train))
print("Batch size: {}".format(cfg.BATCH_SIZE))

num_output = cfg.ANCHOR_PER_GRID * (cfg.CLASSES + 1 + 4)
ESDet = ESDet(cfg, (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.N_CHANNELS), num_output)
ESDet.model.summary()
#ESDet.model.load_weights("checkpoints/model.14-63.09.hdf5")


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5.0e-3,
    decay_steps=1000,
    decay_rate=0.8)
"""
learning_rates = [2.3e-04, 1.8e-04, 1.4e-04, 1.2e-04, 1.0e-04, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 700, 3000]
lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)
"""

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#cfg.LR= 0.0001 



#callbacks
cb = []
if REDUCELRONPLATEAU:
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1,verbose=1,
        patience=5, min_lr=0.0
    )

    cb.append(reduce_lr)

ckp_saver = tf.keras.callbacks.ModelCheckpoint("checkpoints" + "/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=0,
                            save_best_only=True,
                            save_weights_only=True, mode='auto', period=1)
cb.append(ckp_saver)

train_generator = generator_from_data_path(img_names, gt_names, config=cfg, shuffle=True)
validation_generator = generator_from_data_path(img_names_val, gt_names_val, config=cfg)


loss_fn = ESDetLoss(cfg)
box_loss_fn = ESDetBoxLoss(cfg)
class_loss_fn = ESDetClassLoss(cfg)
conf_loss_fn = ESDetConfLoss(cfg)


ESDet.model.compile(
    optimizer=opt,
    loss=[loss_fn], 
    metrics=[box_loss_fn, class_loss_fn, conf_loss_fn]
)

#actually do the training
ESDet.model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    steps_per_epoch=nbatches_train,
    validation_steps=nbatches_val,
    verbose=1,
    callbacks=cb
)

    