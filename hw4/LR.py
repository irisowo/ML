import argparse
import numpy as np
from scipy.special import expit
from numpy.linalg import det, inv
from hw4_lib import data_generator #expit

# ----------------------------------------------
# test case 1 : default                    
# test case 2 : -x1 1 2 -y1 1 2 -x2 3 4 -y2 3 4                         
# ----------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type = int, default=50)
    parser.add_argument('-x1', nargs=2, type=float, default=[1.0, 2.0])  #m_x1, v_x1
    parser.add_argument('-y1', nargs=2, type=float, default=[1.0, 2.0])  #m_y1, v_y1
    parser.add_argument('-x2', nargs=2, type=float, default=[10.0, 2.0]) #m_x2, v_x2
    parser.add_argument('-y2', nargs=2, type=float, default=[10.0, 2.0]) #m_y2, v_y2
    return parser.parse_args()


class LR:
    def __init__(self, N, X, Label):
        # raw data
        self.N = N
        self.X = X
        self.Label = Label
        # w learned from gradient and newton's method
        self.w_grad = np.zeros((3, 1))
        self.w_newtons = np.zeros((3, 1))
    

    from hw4_lib import print_confusion_matrix, plot_result


    def gradient_descent(self):
        iter = 1e4
        w = np.zeros((3, 1))
        while(iter):
            # w' = w + dw
            # dw = ▽F = XT[expit(Xw)-y]
            dw = self.X.T @ (self.Label - expit(self.X @ w))
            if (abs(dw) < 1e-6).all(): 
                break
            iter -= 1
            w += dw
        self.w_grad = w
        return


    def newtons(self):
        n = self.N * 2
        w = np.zeros((3, 1))
        D = np.zeros((n, n))
        iter = 1e4
        while(iter):
            # (1) ▽F = XT[expit(Xw)-y]
            expit_Xw = expit(self.X @ w)
            df = self.X.T @ (self.Label - expit_Xw)
            
            # (2) Hf = (XT)D(X)
            #      D = e^-Xw[i]/(1+e^-Xw[i])^2 
            for i in range(n):
                D[i, i] = expit_Xw[i] @ (1 - expit_Xw[i])
            Hf = self.X.T @ D @ self.X
            
            # (3) dw = ▽F(f), if H(f) is invertible
            #        = H(f)_inv @ ▽F(f), otherwise
            dw = df
            if (det(Hf) != 0):
                dw = inv(Hf) @ df
            
            # termination
            if (abs(dw) < 1e-4).all(): 
                break
            
            # update
            w += dw
            iter -= 1
        
        self.w_newtons = w
        return


if __name__ == '__main__':
    args = get_parser()
    N = args.N
    d1 = data_generator(args.x1, args.y1, N)
    d2 = data_generator(args.x2, args.y2, N)
    X = np.concatenate((d1, d2), axis=0)
    
    Label = np.zeros((2 * N, 1))
    for i in range(N, 2 * N):
        Label[i, 0] = 1
    
    LR_Model = LR(N, X, Label)
    LR_Model.gradient_descent()
    LR_Model.newtons()

    LR_Model.plot_result()
   


    

