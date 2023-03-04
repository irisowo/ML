import numpy as np
import math

# An easy-to-program approximate approach that relies on the central limit theorem is as follows: 
# generate 12 uniform U(0,1) deviates, add them all up, and subtract 6 
# X = μ + σZ, where Z is standard normal.
def Data_Generator_Univar(m, s):
    Z = sum(np.random.uniform(0, 1, 12)) - 6
    return (m + math.sqrt(s) * Z )


# y = W．ø(x) + e
# size(W) = (n,1), e ~ N(0,a)
def Data_Generator_Polyn(n, a, w):
    y = Data_Generator_Univar(0, a) # e
    x = np.random.uniform(-1.0, 1.0)
    for i in range(n):
        y += w[i] * pow(x, i)
    return (x, y)