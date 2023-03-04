import argparse
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import MultipleLocator

#===================================================================================
# (1) Preprocess
def myparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default = "./testfile.txt", help = "filepath")
    parser.add_argument('-n', type=int, default = 2, help = "number of bases")
    parser.add_argument('-l', type=float, default = 0.0, help = "lambda")
    args = parser.parse_args()
    print("Case : n = ", args.n, " lamb = ", args.l)
    return (args.f, args.n, args.l)
    
def read_testfile(testfile):
        X, Y = [], []
        with  open("./"+testfile, "r") as f:
            for line in f:
                x, y = line.split(',')
                X.append(float(x))
                Y.append(float(y))
        return X, Y

def build_design_matrix(X, n):
    lenX = len(X)
    A = np.zeros((lenX, n))
    for i in range(lenX):
        xi = X[i]
        A[i] = [xi**power for power in range (n-1, -1, -1)]
    return A
#===================================================================================
# (2) Matrix operations
def dot_product(a, b):
    product = 0.0
    for i in range (len(a)):
        product += a[i]*b[i]
    return product

def mat_trans(M):
    row, col = M.shape[0], M.shape[1]
    MT = np.zeros((col, row))
    for i in range(row):
        for j in range(col):
             MT[j,i] = M[i,j]
    return MT
    
def matmul(M, N):
    row_M, col_M, col_N = M.shape[0], M.shape[1], N.shape[1]
    MN = np.zeros((row_M, col_N))
    for i in range(row_M):
        for j in range(col_N):
            for k in range(col_M):
                MN[i,j] += M[i,k] * N[k,j]
    return MN
#===================================================================================
# (3) LSE
def AT_mul_A_add_lambI(AT, A, n, lamb):
    ATA_lambI = matmul(AT, A)
    for i in range(n):
        ATA_lambI[i,i] += lamb
    return ATA_lambI

def AT_mul_b(AT, b, n):
    ATb = np.zeros((n,1))
    for i in range(n):
        ATb[i,0] = dot_product(AT[i], b)
    return ATb
#===================================================================================
# (4) Result & plot
def print_result(A,b,W,n):
    print("Fitting line:", end=' ')
    for i in range(n):
        wi = W[i,0]
        if wi == 0.0: continue
        elif i == 0 or wi < 0.0: 
            print(format(W[i][0], '.12g'), end = '')
        else:
            print("+",format(W[i][0], '.12g'), end='')
        
        if i < n-1:
            print("X^"+str(n-1-i), end=' ')

    # Error AW-B
    Err =  0.0
    for i in range(len(A)):
        Err += (dot_product(A[i], W)-b[i])**2
    print("\nTotal error: ", format(Err[0], '.12g'))
    return


def plot(LSE_w, Newton_w, X, Y, n):
    min_x, max_x = min(X), max(X)
    min_y, max_y = min(Y),max(Y)

    #_x = np.linspace(min_x-2.0,max_x+2.0)
    _x = np.arange(min_x - 1.0, max_x + 1.0, step=0.01)
    LSE_y, Newton_y = 0.0, 0.0
    for i in range(n):
        power = n-1-i
        LSE_y += LSE_w[i]*(_x ** power)
        Newton_y += Newton_w[i]*(_x ** power)
        
    X_interval = MultipleLocator(2)
    Y_interval = MultipleLocator(20)

    plt.figure(1)
    plt.subplot(211) #2x1-1
    plt.scatter(X, Y, c='r',edgecolors='k')
    plt.plot(_x,LSE_y, c='k')
    ax = plt.gca()
    ax.xaxis.set_major_locator(X_interval)
    ax.yaxis.set_major_locator(Y_interval)
    plt.axis([min_x - 1, max_x + 1, min_y - 10, max_y + 10])
     
    plt.subplot(212) #2x1-2
    plt.scatter(X, Y, c='r', edgecolors='k')
    plt.plot(_x,Newton_y, c='k')
    ax = plt.gca()
    ax.xaxis.set_major_locator(X_interval)
    ax.yaxis.set_major_locator(Y_interval)
    plt.axis([min_x - 1, max_x + 1, min_y - 10, max_y + 10])

    plt.show()
    return