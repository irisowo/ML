import math
import numpy as np
import gzip, struct, binascii

NUM_CLASSES = 10
IMG_ROWS = 28
IMG_COLS = 28

filepath = "./"
train_img_path = filepath + "gz/train-images-idx3-ubyte.gz"
train_label_path = filepath + "gz/train-labels-idx1-ubyte.gz"


#--------------------------------------------------------------------------------------#
# (1) read_labels, read_images                                                         #
#--------------------------------------------------------------------------------------#
def read_labels(label_path):
    with gzip.open(label_path, 'r') as f:
        magic, cnt_labels = struct.unpack(">2I", f.read(8)) # Big indian & unsigned int
        label_data = f.read() # unsigned byte (0~9)
        labels = np.frombuffer(label_data, dtype=np.uint8, count=cnt_labels) 
    f.close
    return labels


def read_images(img_path, label_path):
    # paths of train data
    labels = read_labels(label_path)
    labels = np.asarray(labels)
    
    with gzip.open(img_path, 'r') as f:
        magic, cnt_imgs, rows, cols = struct.unpack(">4I", f.read(16)) 
        img_size = rows * cols  
        
        images = [ [0] * (img_size) for i in range(cnt_imgs)]
        for i in range(cnt_imgs):
            for j in range(img_size):
                #gray = int(struct.unpack(">c", f.read(1))) # unsigned byte(0~255)
                gray = int(binascii.b2a_hex(f.read(1)), 16)
                if(gray > 127):
                    images[i][j] = 1
                else:
                    images[i][j] = 0
        images = np.reshape(images, (cnt_imgs, img_size))

    return images, labels


#--------------------------------------------------------------------------------------#
# (2) assign_label, print_imagination, print_confusion                                 #
#--------------------------------------------------------------------------------------#
def assign_label(cnt_predict):
    # Marginize confusion -> P_label
    P_label = np.zeros((10, 10))
    for i in range(NUM_CLASSES):
        P_label[i, :] = cnt_predict[i, :]/np.sum(cnt_predict[i, :])
    P_label = P_label.ravel()

    class_each_label = np.full(10, -1)
    assigned_class = np.full(10, False)
    i = 0
    while i < 10:
        # assign from the maximum p_label
        tmp = np.argmax(P_label) #np.argmax return an idx
        if P_label[tmp] == 0:
            break

        # la: label, c: class
        la = tmp // 10
        c = tmp % 10
        if assigned_class[c] == False and class_each_label[la] == -1:
            class_each_label[la] = c
            assigned_class[c] = True
            i += 1

        # mark the entry 0 after assignment
        P_label[tmp] = 0
    return class_each_label


def print_imagination(P_class, logging, label):
    imagination = (P_class >= 0.5)*1
    for i in range(NUM_CLASSES):
        label_idx = label[i]
        print(logging.format(i))
        for row in range (IMG_ROWS):
            for col in range(IMG_COLS):
                idx = row * IMG_COLS + col
                print(imagination[label_idx][idx], end=' ')
            print()
        print()


def print_confusion(confusion, label, iter):
    error = 60000
    for i in range(NUM_CLASSES):
        tp = confusion[i, label[i]]
        fp = np.sum(confusion[i])-tp
        fn = np.sum(confusion[:,label[i]])-tp
        tn = 60000-tp-fp-fn
        error -= tp

        print("---------------------------------------------------------\n")
        print("Confusion Matrix :", i)
        print("\t\tPredict number ", i, " Predict not number ",i)
        print("Is numbr {i}\t\t{tp}\t\t{fn}".format(i=i, tp=tp, fn=fn))
        print("Isn't numbr {i}\t\t{fp}\t\t{tn}".format(i=i, fp=fp, tn=tn))
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print("\nSensitivity (Successfully predict number ", i, " : {:7.5f}".format(sensitivity))
        print("Specificity (Successfully predict not number ", i, " : {:7.5f}\n".format(specificity))
        
    print('Total iteration to converge:', iter)
    print('Total error rate:', error/60000)