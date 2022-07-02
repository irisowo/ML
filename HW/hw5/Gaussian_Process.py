import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
beta = 5.0
points = 6000


def read_input_data():
    data = np.loadtxt('./data/input.data', dtype=float)
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)
    return X, Y

def k_RQ(xi, xj, l, a):
    # l : length scale with l > 0
    # a(ahpla) : scale mixture param with α > 0
    # (1 + (xi - xj)^2) / (2αl^2)^(-α)
    return (1 + cdist(xi, xj, 'sqeuclidean')/(2 * a * (l ** 2))) ** (-a)

def cov(l, alpha):
    cov_matix = k_RQ(X, X, l, alpha)
    noise = (1 / beta) * np.identity(X.shape[0])
    return (cov_matix + noise)

def objective_function(theta):
    # Objective function J = -ln(p(y|θ) = 0.5[Nln(2π)+ yT(C−^1)y + ln|Cθ|]
    # θ = (l, α)
    N = X.shape[0]
    YT = Y.ravel().T
    C = cov(theta[0], theta[1])
    
    obj_f = 0.5 * (N * np.log(2 * np.pi) +
                (YT @ np.linalg.inv(C) @ Y) +
                np.log(np.linalg.det(C)))
    return obj_f[0]

def visualization(mean, var):
    # z = ±1.96 = (x-m)/var => x = m ±1.96 * var
    upper = mean + 1.96 * var.diagonal().reshape(-1, 1)
    lower = mean - 1.96 * var.diagonal().reshape(-1, 1)
    
    plt.xlim(-60, 60)
    plt.ylim(max(upper) + 2, min(lower) - 2)
    plt.title("Part2")
    # Mark 95% interval
    plt.fill_between(Xtest, upper.ravel(), lower.ravel(), color='thistle')
    # Draw a line to represent mean of f
    plt.plot(Xtest, mean, 'royalblue')
    # Show all training data points
    plt.scatter(X, Y, s=10, c='r')
    plt.show()

def Gaussian_Process(l=1.0, alpha=1.0):
    # compute C, k(x, x*), and k*
    C = cov(l, alpha)
    k_x_xstar = k_RQ(X, Xstar, l, alpha)
    kstar = k_RQ(Xstar, Xstar, l, alpha) + 1 / beta
    
    # p(y*|y) ~ N(μ, σ^2) 
    mean = k_x_xstar.T @ (np.linalg.inv(C)) @ Y 
    var = kstar - (k_x_xstar.T @ (np.linalg.inv(C) @ k_x_xstar))

    # plot the figure
    visualization(mean, var)
    return

if __name__ == '__main__':
    X, Y = read_input_data()
    
    # generate X* and its 1d version
    Xstar = np.linspace(-60, 60, points).reshape(-1, 1)
    Xtest = Xstar.ravel()

    Gaussian_Process()
    
    # Let's initialize θ by the params(l=1, α=1) in part1
    theta = [1.0, 1.0]
    theta_new = minimize(objective_function, theta)
    print(theta_new.x[0], theta_new.x[1])
    Gaussian_Process(theta_new.x[0], theta_new.x[1])