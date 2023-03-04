from hw1_lib import *
import numpy as np
# cd ./Desktop/ML/HW/hw1

# ref : https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html
def LU(M):
    U = M.copy()
    L = np.zeros((n,n))
    for i in range(n): L[i,i] = 1.0

    #  If EA = U, then E_inv = L since (E_inv)EA = LU
    for i in range(n): # pivots
        factor = U[i+1:, i] / U[i, i] # multiple of downward row 
        L[i+1:, i] = factor #reverse the row operations to manipulate L
        U[i+1:] -= factor[:, np.newaxis] * U[i] #Eliminate entries below i with row operations on U
    return L,U

def LU_decomposition(M,b):
    # Goal : Given LUx = b, find x
    L, U = LU(M)
    n = len(L)

    # forward_substitution : Ly=b
    y = np.zeros_like(b)
    y[0] = b[0] / L[0, 0] #Initialize the 1st row.
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]

    # back_substitution : Ux = y
    x = np.zeros_like(y)
    x[-1] = y[-1] / U[-1, -1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]
    return x 

def rLSE(A, b, n, lamb):
    #=============================================================
    # LSE  : Ax = b => ATAx = ATb =>  x = Inv(ATA)ATb            #
    # rLSE : minimize |Ax-b|^2 + lamd|x|^2 by differentiation    #
    #       => ATAx + lambx = ATb                                #
    #       => x = Inv(ATA+lambI)ATb                             #
    #=============================================================
    print("LSE:")
    AT = mat_trans(A)
    ATb = AT_mul_b(AT, b, n)
    ATA_lambI = AT_mul_A_add_lambI(AT, A, n, lamb)
    W = LU_decomposition(ATA_lambI, ATb) # (ATA+lamdx)W = ATb
    print_result(A,b,W,n)
    return W

def Newtons_Method(A, b, n):
    #=============================================================
    # Newton's Method                                            #
    # f(x) =||Ax-b||^2 = xTATAx - 2ATb + bTb                     #
    # Thus, ∇f(x) = 2ATAx - 2ATb ; H_f(x) = 2ATA                 #
    # dx = (H_fx)_inverse * ∇fx                                  #
    # x1 = x0 - dx                                               #
    #=============================================================
    print("Newton's Method:")
    AT = mat_trans(A)
    ATb = AT_mul_b(AT, b, n)
    x0 = np.zeros((n,1))
    H_fx = 2 * matmul(AT,A)
    Gradient_fx = matmul(H_fx, x0) - 2 * ATb
    dx = LU_decomposition(H_fx,Gradient_fx) # (H_fx)dx = ∇fx  
    x1 = x0 - dx
    print_result(A,b,x1, n)
    return x1


if __name__ == '__main__':

    testfile, n, lamb = myparser()
    X, Y = read_testfile(testfile)
    A = build_design_matrix(X, n)
    b = np.array(Y)[:, np.newaxis]
    
    LSE_w = rLSE(A, b, n, lamb)
    print()
    Newtons_w = Newtons_Method(A, b, n)

    plot(LSE_w , Newtons_w, X, Y, n)


    
