import os, math
import argparse
from hw2_lib import *


# Usage : python ./hw2-1.py --m [0/1]
# Note that train and test file should be placed under directory names "gz"
def parse_args():
    parser = argparse.ArgumentParser(description = "0 for Discrete and 1 for Continuous")
    parser.add_argument("--m", default = 0, type = int)
    return parser.parse_args()

def Discrete_mode(test_images, test_labels, frequency, prior):
    cnt_error = 0
    img_size = len(test_images[0])

    for i in range(NUM_TEST):
        # ln(prior * likelihood) = ln(prior) + ln(likelihood)
        posterior = np.log(prior) # len(posterior) = 10
        for label in range (NUM_CLASSES):
            for j in range(img_size):
                bin = test_images[i][j]//8
                posterior[label] += np.log(frequency[label][j][bin])
        
        # Marginalization
        posterior /= np.sum(posterior)

        # print the posterior of each label and make prediction
        cnt_error += print_MAP_prediction(posterior, test_labels[i])

    # return error rate 
    return cnt_error/NUM_TEST

def Continuous_mode(test_images, test_labels, mean, prior):
    cnt_error = 0
    img_size = len(test_images[0])
    var = np.load("./var.npy")

    # ln(Gaussin) = (-1/2)ln(2 PI var) + -(x-mu)^2/2var
    for i in range(NUM_TEST):
        posterior = np.log(prior)
        for label in range (NUM_CLASSES):
            for j in range(img_size):
                var_pixel = var[label][j]
                # Note that lg(0) and 1/0 don't exist
                if var_pixel == 0: 
                    continue
                posterior[label] -= 0.5 * np.log(2.0 * math.pi * var_pixel)
                posterior[label] -= ((test_images[i][j] - mean[label][j])**2) / (2.0 * var_pixel)
        
        # Marginalization
        posterior /= sum(posterior)
    
        # print the posterior of each label and make prediction
        cnt_error += print_MAP_prediction(posterior, test_labels[i])
    
    # return error rate
    return cnt_error/NUM_TEST

#--------------------------------------------------------------------------
if __name__ == '__main__':

    args = parse_args()
    MODE = args.m

    # Check whether frequency.npy, mean.npy, and var.npy exist
    TRAINED = ((os.path.isfile("./prior.npy") and os.path.isfile("./frequency.npy")) and 
                (os.path.isfile("./mean.npy") and os.path.isfile("./var.npy")))
    
    # load images and labels
    if (not TRAINED) : 
        read_train_data() # prior, frequency, mean, and val are created by read_train_data()
        
    # load test images and labels
    test_image, test_label = read_test_data()
    
    # start prediction
    prior = np.load("./prior.npy")
    if MODE == 0:
        frequency = np.load("./frequency.npy")
        error_rate = Discrete_mode(test_image, test_label, frequency, prior)
        print_imagination(frequency, '0')
        print("Error rate: {:.4f}".format(error_rate))
    elif MODE == 1:
        mean = np.load("./mean.npy")
        error_rate = Continuous_mode(test_image, test_label, mean, prior)
        print_imagination(mean, '1')
        print("Error rate: {:.4f}".format(error_rate))