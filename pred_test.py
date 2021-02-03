import tensorflow as tf
from src.dataGenerator import generator_from_data_path
from model.ESDet import ESDet
from src.create_config import load_dict
from tensorflow.keras import backend as K
import numpy as np
from src.evaluation import filter_batch
import cv2
import matplotlib.pyplot as plt
from src.losses import *
import src.utils_squeezeDet as utils
import itertools 



from src.visualization import visualize
#from src.losses import loss, loss_without_regularization, bbox_loss, class_loss, conf_loss


img_file = "img_train.txt"
gt_file = "gt_train.txt"
img_file_val = "img_val.txt"
gt_file_val = "gt_val.txt"
EPOCHS = 5
STEPS = None
REDUCELRONPLATEAU = True


#open files with images and ground truths files with full path names
with open(img_file_val) as imgs:
    img_names_val = imgs.read().splitlines()
imgs.close()
with open(gt_file_val) as gts:
    gt_names_val = gts.read().splitlines()
gts.close()

config_file= "squeeze.config"
cfg = load_dict(config_file)
cfg.EPOCHS = EPOCHS
cfg.BATCH_SIZE = 1



nbatches_val, mod = divmod(len(img_names_val), cfg.BATCH_SIZE)



num_output = cfg.ANCHOR_PER_GRID * (cfg.CLASSES + 1 + 4)
ESDet = ESDet(cfg, (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.N_CHANNELS), num_output)
ESDet.model.summary()


# create sgd with momentum and gradient clipping
opt =  tf.keras.optimizers.Adam(clipnorm=cfg.MAX_GRAD_NORM)
cfg.LR= 0.00001 


print("Learning rate: {}".format(cfg.LEARNING_RATE))

#callbacks
cb = []
if REDUCELRONPLATEAU:
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1,verbose=1,
        patience=5, min_lr=0.0
    )

    cb.append(reduce_lr)


#train_generator = generator_from_data_path(img_names, gt_names, config=cfg)
validation_generator = generator_from_data_path(img_names_val, gt_names_val, config=cfg)



#plt.figure()
#plt.imshow(img)
#plt.title("ghdghyh")
#plt.axis('off')
#plt.show()




loss_fn = ESDetLoss(cfg)
box_loss_fn = ESDetBoxLoss(cfg)
class_loss_fn = ESDetClassLoss(cfg)
conf_loss_fn = ESDetConfLoss(cfg)

ESDet.model.compile(
    optimizer=opt,
    loss=[loss_fn], 
    metrics=[box_loss_fn, class_loss_fn, conf_loss_fn]
)

ESDet.model.load_weights("checkpoints/model.07-1.27.hdf5")



def bbox_transform_single_box(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = int(np.floor(cx-w/2))
    out_box[1] = int(np.floor(cy-h/2))
    out_box[2] = int(np.floor(cx+w/2))
    out_box[3] = int(np.floor(cy+h/2))

    return out_box

def draw_gt(y_true, font, img):
    box_input = y_true[:, :, 1:5]
    labels = y_true[:, :, 9:]
    non_zero_boxes = box_input[0][box_input[0] > 0].reshape((-1,4))

    non_zero_labels = []
    #get the predicted labels
    for k, coords in enumerate(box_input[0,:]):
        if np.sum(coords) > 0:

            for j, l in enumerate(labels[0, k]):
                if l == 1:
                    non_zero_labels.append(j)

    #iterate gt boxes
    for j, gt_box in enumerate(non_zero_boxes):

        #transform into xmin, ymin, xmax, ymax
        gt_box = bbox_transform_single_box(gt_box)

        #add rectangle and text
        cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 1)
        cv2.putText(img, cfg.CLASS_NAMES[int(non_zero_labels[j])], (gt_box[0], gt_box[1]), font, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
        
    return img


def draw_pred(y_pred, font, img):
    #get predicted boxes
    all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(y_pred, cfg)

    #print( np.array(all_filtered_boxes).shape)

    #iterate predicted boxes
    for j, det_box in enumerate(all_filtered_boxes[0]):

        #transform into xmin, ymin, xmax, ymax
        det_box = bbox_transform_single_box(det_box)

        #add rectangle and text
        cv2.rectangle(img, (det_box[0], det_box[1]), (det_box[2], det_box[3]), (0,0,255), 1)
        cv2.putText(img, cfg.CLASS_NAMES[all_filtered_classes[0][j]] + " " + str(all_filtered_scores[0][j]) , (det_box[0], det_box[1]), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

    return img

for i in range(20, 40):
    
    imgs, y_true = next(itertools.islice(validation_generator, i, None))
    y_pred = ESDet.model.predict(imgs)

    img = cv2.imread(img_names_val[i])
    img = cv2.resize( img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
    font = cv2.FONT_HERSHEY_SIMPLEX

    img = draw_gt(y_true, font, img)
    img = draw_pred(y_pred, font, img)

    cv2.imshow("text", img)
    cv2.waitKey(0)  
    
    #closing all open windows  
    cv2.destroyAllWindows()  