import argparse
import numpy as np
from hw3_lib import design_matrix
from DataGenerator import Data_Generator_Polyn

# ------------------------------------------
# test case 1 : default                    
# test case 2 : -b 100                     
# test case 3 : -b 1 -n 3 -a 3 -w 1 2 3      
# ------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type = float, default=1.0) #100
    parser.add_argument('-n', type = int, default=4)
    parser.add_argument('-a', type = float, default=1.0)
    parser.add_argument('-w', nargs='+', type=float, default=[1.0, 2.0, 3.0, 4.0])
    return parser.parse_args()


class BayesianLinearRegression:    
    def __init__(self, b:float, n:int, a:float, w: np.ndarray):
        self.b = b
        self.n = n
        self.a = a
        self.w = w
        
        # record (x, y) and (mean, var) of specific iter
        self.x, self.y = [], []
        self.mean, self.var = [], []

        # initialize prior
        self.prior_mean = np.zeros((n, 1))
        self.prior_cov = b * np.identity(n)
        
        # initialize posterior
        self.post_mean = np.zeros((n, 1))
        self.post_cov = np.identity(n)
        self.post_var = np.identity(n)
    
    
    # Import visualization functions
    from hw3_lib import visualization, print_post_mean_var


    def generate_point(self):
        x, y = Data_Generator_Polyn(self.n, self.a, self.w)
        self.x.append(x), self.y.append(y)
        print("Add data point ({:.5f}, {:.5f}):\n".format(x, y))
        return x, y


    def update_post_mean_var(self, X, y):
        # Caculate posterior
        #   Λ = aXTX + S
        #   μ = Λ_inv(aXTY + Sm)
        #   For the 1st iter : 
        #       Λ = aXTX + bI
        #       μ = aΛ_inv(XTY) = Λ_inv(aXTY + Sm) if we asuume m = 0
       
        # Λ = aXTX + S
        a = self.a # a
        self.post_cov = a * (X.T @ X) + self.prior_cov
        self.post_var = np.linalg.inv(self.post_cov)
        
        # μ = Λ_inv(aXTY + Sm)
        Sm = self.prior_cov @ self.prior_mean
        self.post_mean = self.post_var @ (a * X.T * y + Sm)
        self.print_post_mean_var()
    

    def predict_distribution(self, X):
        # predictive distribution :
        #   N(Xμ, 1/a + X Λ_inv XT)
        a = self.a # a
        varinv_XT = np.linalg.inv(self.post_cov) @ X.T
        predict_mean = (X @ self.post_mean)[0, 0]
        predict_var = (1 / a + X @ varinv_XT)[0, 0]
        print("Predictive distribution ~ N({:.5f}, {:.5f})\n".format(predict_mean, predict_var))


    def update_prior(self):
        self.prior_cov = self.post_cov
        self.prior_mean = self.post_mean


    def Baysian_LR(self):
        iter = 0
        while True:
            x, y = self.generate_point()
            X = design_matrix(x, n)

            self.update_post_mean_var(X, y)
            self.predict_distribution(X)

            if iter == 10 or iter == 50:
                self.var.append(self.post_var)
                self.mean.append(self.post_mean)
            
            error = self.post_mean - self.prior_mean
            if (abs(error) < 0.0001).all() and iter > 50:
                self.var.append(self.post_var)
                self.mean.append(self.post_mean)
                break

            self.update_prior()
            iter += 1
        return

    
if __name__ == '__main__':
    args = get_parser()
    b, n, a, w= args.b, args.n, args.a, args.w

    BLR = BayesianLinearRegression(b, n, a, w)
    BLR.Baysian_LR()
    BLR.visualization()