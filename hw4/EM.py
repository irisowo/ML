from hw4_lib2 import *


class EM:
    def __init__(self, train_imgs, train_labels):
        # raw data
        self.X = train_imgs
        self.labels = train_labels
        self.cnt_imgs = train_imgs.shape[0]
        self.img_size = train_imgs.shape[1]
        
        self.P = np.random.uniform(0.0, 1.0, (NUM_CLASSES, self.img_size))
        self.P /= np.sum(self.P)

        self.lamb = np.full(NUM_CLASSES, 0.1)
        self.w = np.zeros((self.cnt_imgs, NUM_CLASSES))
      

    def set_params(self, w, P, lamb):
      self.w = w
      self.P = P
      self.lamb = lamb


    #---------------------------------------------------------
    # E-step:                                                #
    #   w : p(label|X)                                       #
    #   w = p(z=label, X) / marginal                         #
    #     = λ * Π p(x_pixel) / marginal                      #
    #---------------------------------------------------------
    def E_step(self):
        w = np.zeros((self.cnt_imgs, NUM_CLASSES))
        for i in range(self.cnt_imgs):
            for j in range(NUM_CLASSES):
                # w = λ * Π p(x_pixel)
                w[i, j] = self.lamb[j]
                for k in range (self.img_size):
                    p = self.P[j, k]
                    w[i, j] *= p if self.X[i, k] else (1 - p)
            
            # normalization 
            w_sum = np.sum(w[i, :])
            if w_sum:
                w[i, :] /= w_sum
        self.w = w
        return w

    
    #---------------------------------------------------------
    # M-step:                                                #
    #   w : p(Z|X)                                           #
    #   λ : Σw/marginal                                      #
    #   P = Σw * xi / Σw                                     #
    #---------------------------------------------------------
    def M_step(self):
        P = np.zeros((NUM_CLASSES, self.img_size))
        sum_wi = np.sum(w, axis=0)

        for i in range(NUM_CLASSES):
            for j in range(self.img_size):
                for k in range(self.cnt_imgs):
                    P[i, j] += w[k, i] * self.X[k, j]
                P[i, j] = (P[i, j] + 1e-8) / (sum_wi[i] + 1e-8 * 784) # img_size = 784
          
        self.P = P
        self.lamb = (sum_wi + 1e-8)/(np.sum(sum_wi) + 1e-8 * 10)
        return P, self.lamb

    
    #--------------------------------------------------------
    # The output will be like (just an example):            #
    #                                                       #
    #       class0  class1  class2  ... class9              #
    # lab 0   10     [200]    10          10                #
    # lab 1  [300]     10     10          10                #
    #  ...                                                  #
    # lab9    1        1       1         [300]              #
    #                                                       #
    #--------------------------------------------------------
    def predict_label(self):
        P_label = np.zeros(NUM_CLASSES)
        cnt_predict = np.zeros((NUM_CLASSES, NUM_CLASSES))

        for i in range(self.cnt_imgs):
            for j in range(NUM_CLASSES):
                P_label[j] = lamb[j]
                for k in range (self.img_size):
                    p = self.P[j, k]
                    P_label[j] *= p if self.X[i, k] else (1 - p)
            # predict += 1 based on MAP
            cnt_predict[self.labels[i], np.argmax(P_label)] += 1
        return cnt_predict


    #--------------------------------------------------------
    # The output are:                                       #
    # (1) Imagination of final prediction                   #
    # (2) confusion matrix and total error rate             #
    #--------------------------------------------------------
    def print_result(self, cnt_predict, iter):
        label = assign_label(cnt_predict)
        print_imagination(self.P, "labeled class {}:", label)
        print_confusion(cnt_predict, label, iter)


train_mode = False
if __name__ == '__main__':

    # 1. load data
    train_imgs, train_labels = read_images(train_img_path, train_label_path)
    EM_Model = EM(train_imgs, train_labels)
    P = np.copy(EM_Model.P)
    w = np.copy(EM_Model.w)
    lamb = np.copy(EM_Model.lamb)

    # 2. loop
    iter = 18
    
    while True:
        # params
        iter += 1
        P_prev = np.copy(EM_Model.P)
        w_path = (filepath + "npy/w"+str(iter) + ".npy")
        p_path = (filepath + "npy/p" + str(iter) + ".npy")
        lamb_path = (filepath + "npy/lambda" + str(iter) + ".npy")
        
        # calculate the params : w, P , lamb
        if train_mode:
            w = EM_Model.E_step()
            P, lamb = EM_Model.M_step()

            np.save(w_path, w)
            np.save(p_path, P)
            np.save(lamb_path, lamb)
        
        else:
            w = np.load(w_path)
            P = np.load(p_path)
            lamb = np.load(lamb_path)

        # params
        diff = np.linalg.norm(P - P_prev)
        default_label = [i for i in range(10)]
        
        # logging
        print_imagination(P, "class {}:", default_label)
        print("No. of Iteration: ", iter, ", Difference: {}\n".format(diff))
        print("---------------------------------------------------------\n")
        
        # convergence criteria
        if iter == 20 or diff < 1e-1:
            break
    
    # 3. make final prediction and print the result
    # cnt the prediction
    EM_Model.set_params(w, P, lamb)
    cnt_predict = EM_Model.predict_label()
    
    # print final result
    EM_Model.print_result(cnt_predict, iter)