from socket import if_nametoindex
import numpy as np
import gzip, struct, binascii

# Usage : python ./hw2-1.py --> input(mode)
# Note that train and test file should be placed under directory names "gz"

NUM_TEST = 100; # modify this

NUM_CLASSES = 10
IMG_ROWS = 28
IMG_COLS = 28

#--------------------------------------------------------------------------
# (1) Read file function :  read_labels, read_train_data(), read_test_data()
def read_labels(label_path):
    with gzip.open(label_path, 'r') as f:
        magic, cnt_labels = struct.unpack(">2I", f.read(8)) # Big indian & unsigned int
        label_data = f.read() # unsigned byte (0~9)
        labels = np.frombuffer(label_data, dtype=np.uint8, count=cnt_labels) 
    f.close
    return labels

def read_train_data(): 
    label_path = "./gz/train-labels-idx1-ubyte.gz"
    image_path = "./gz/train-images-idx3-ubyte.gz"

    # read labels   
    labels = read_labels(label_path)

    # read images
    with gzip.open(image_path, 'r') as f:
        # format : Big-endian, unsgined int
        magic, cnt_imgs, rows, cols = struct.unpack(">4I", f.read(16)) 
        img_size = rows * cols    

        # declaration
        images = [ [0]*(img_size) for i in range(cnt_imgs)]
        frequency = np.zeros((NUM_CLASSES, img_size, 32), dtype=float)
        prior = np.zeros(NUM_CLASSES, dtype=float)
        mean = np.zeros((NUM_CLASSES, img_size), dtype=float)
        var = np.zeros((NUM_CLASSES, img_size), dtype=float)
       
        # read images and record the likelihood, mean
        for i, label in enumerate(labels):
            prior[label] += 1.0
            for j in range(img_size):
                #gray = int(struct.unpack(">c", f.read(1))) # unsigned byte(0~255)
                gray = int(binascii.b2a_hex(f.read(1)), 16)
                images[i][j] = gray
                frequency[label, j, (gray//8)] += 1.0
                mean[label][j] += images[i][j]

        # marginalize mean and frequency(likelihood)
        total_num = frequency.sum(2) # sum(label, j)
        for label in range(NUM_CLASSES):
            mean[label, :] /= prior[label] # prior has not been marginalized, so it stores the count of label now.
            for j in range(img_size):
                frequency[label][j] /= total_num[label][j]
                
        frequency[frequency == 0.0] = 0.0001 #pesudocount
        np.save("./frequency.npy", frequency)
        np.save("./mean.npy", mean)

        # get variance
        for i, label in enumerate(labels):
            for j in range(img_size):
                var[label][j] += (images[i][j] - mean[label][j])**2
        for label in range(NUM_CLASSES):
            var[label][:] /= prior[label]
        np.save("./var.npy", mean)

        # marginalize prior
        prior /= cnt_imgs
        np.save("./prior.npy", prior)
    f.close

    return

def read_test_data():    
    label_path = "./gz/t10k-labels-idx1-ubyte.gz"
    image_path = "./gz/t10k-images-idx3-ubyte.gz"
    # read labels
    labels = read_labels(label_path)

    # read images
    with gzip.open(image_path, 'r') as f:
        magic, cnt_images, rows, cols = struct.unpack(">IIII", f.read(16)) 
        img_size = rows*cols
        #-------------------modify-----------------#
        cnt_images = NUM_TEST
        #-------------------modify-----------------#
        images = [ [0]*(img_size) for i in range(cnt_images)]
        for i in range(cnt_images):
            for j in range(img_size):
                #gray = int(struct.unpack(">c", f.read(1))) # unsigned byte (0~255)
                gray = int(binascii.b2a_hex(f.read(1)), 16)
                images[i][j] = gray
    f.close
    return images, labels

#--------------------------------------------------------------------------
# (2) Logging function : print_posterior_prediction(), print_imagination
def print_MAP_prediction(posterior, test_label_i):
    print("Postirior (in log scale):")
    for label in range(NUM_CLASSES):
        print(str(label)+":", posterior[label])
    
    # Note that log(prior * likelihood) should be negative, while it's positive now after marginalization
    predict = np.argmin(posterior)
    print("Prediction: ", predict, "Ans:", test_label_i, end = "\n\n")
    
    # return cnt(1 or 0) of prediction error
    return 1 * (predict != test_label_i)

def print_imagination(images, mode):
    print("Imagination of numbers in Bayesian classifier:\n")
    img_size = len(images[0])
    for i in range(NUM_CLASSES):
        print("{}:".format(i))
        for row in range (IMG_ROWS):
            for col in range(IMG_COLS):
                idx = row * IMG_COLS + col
                if mode == '0':
                    cnt_white = sum(images[i, idx, 0:16])
                    cnt_black = sum(images[i, idx, 16:32])
                    print(1*(cnt_white <= cnt_black), end=' ')
                else:
                    print(1*(images[i, idx] >= 128), end=' ')
            print()
        print()
    return