import math
import numpy as np
import matplotlib.pyplot as plt

def data_generator_univar(m, s):
    Z = sum(np.random.uniform(0, 1, 12)) - 6
    return(m + math.sqrt(s) * Z )


def data_generator(x, y, N):
    # X = [x, y, 1] 
    # since f(x, y) = ax + by + c = Xw
    data = np.zeros((N, 3))
    mx, vx = x[0], x[1]
    my, vy = y[0], y[1]
    for i in range(N):
        data[i, 0] = data_generator_univar(mx, vx)
        data[i, 1] = data_generator_univar(my, vy)
        data[i, 2] = 1.0
    return data


def print_confusion_matrix(self, w):
    # print w
    print("w:")
    for i in range(w.shape[0]):
        print("{:15.10f}".format(w[i, 0]))
    print()
        
    # print confusion matrix
    Xw = self.X @ w
    c1, c2 = [], []
    tp, fp, tn, fn = 0.0, 0.0, 0.0, 0.0
    cnt = 1
    for i in range(Xw.shape[0]):
        threshold = Xw[i, 0]
        label = self.Label[i, 0]
        if(threshold < 0):
            if (label == 0):
                tp += 1
            else:
                fp += 1
            c1.append(i)
        else:
            if (label == 1):
                tn += 1
            else:
                fn += 1
            c2.append(i)
    print("Confusion Matrix:")
    print("\t\tPredict cluster 1 Predict cluster 2")
    print("Is cluster 1\t\t{tp}\t\t{fn}".format(tp=tp, fn=fn))
    print("Is cluster 2\t\t{fp}\t\t{tn}\n".format(fp=fp, tn=tn))
    print("Sensitivity (Successfully predict cluster 1): {:7.5f}".format(tp / (tp + fn)))
    print("Specificity (Successfully predict cluster 2): {:7.5f}\n".format(tn / (tn + fp)))

    return c1, c2


def plot_result(self):
    print("Gradient descent:\n")
    grad_c1, grad_c2 =  print_confusion_matrix(self, self.w_grad)
    print("--------------------------------------------------")
    print("Newton's method:\n")
    newtons_c1, newtons_c2 =  print_confusion_matrix(self, self.w_newtons)
    
    N = self.N
   
    plt.subplot(131)
    plt.title("Ground truth")
    c1, c2 = self.X[:N], self.X[N:]
    plt.scatter(c1[:,0], c1[:,1], c='r')
    plt.scatter(c2[:,0], c2[:,1], c='b')

    plt.subplot(132)
    plt.title("Gradient descent")
    c1, c2 = self.X[grad_c1], self.X[grad_c2]
    plt.scatter(c1[:,0], c1[:,1], c='r')
    plt.scatter(c2[:,0], c2[:,1], c='b')


    plt.subplot(133)
    plt.title("Newton's method:")
    c1, c2 = self.X[newtons_c1], self.X[newtons_c2]
    plt.scatter(c1[:,0], c1[:,1], c='r')
    plt.scatter(c2[:,0], c2[:,1], c='b')
    plt.show()

    
    return

'''
def expit(x):
    # logistic function : 1 / (1 + exp(x))
    n = x.shape[0]
    data = np.zeros((n, 1))
    for i in range(n):
        if (x[i, 0] > 30):
            data[i, 0] = 1.0
        elif (x[i, 0] < -30):
            data[i, 0] = 0.0
        else:
            data[i, 0] = 1 / (1 + math.exp(-x[i, 0]))
    return data
'''