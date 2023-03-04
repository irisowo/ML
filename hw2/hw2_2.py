import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default = 0, type = int)
    parser.add_argument("--b", default = 0, type = int)
    return parser.parse_args()

def C(N,m):
    c = math.factorial(N) / (math.factorial(m) * math.factorial(N-m)) 
    return c

def binomial(a, b):
    N = a + b + 0.0
    p = a / N
    bp = C(N,a) * (p**a) * (1.0-p)**b
    return bp

def counter(line):
    cnt_a = line.count('1')
    cnt_b = line.count('0')
    return cnt_a, cnt_b

if __name__ == '__main__':

    args = parse_args()
    prior_a, prior_b = args.a, args.b
    f = "./testfile.txt"
    fp = open(f, "r")
   
    lines = fp.readlines()
    for i, _line in enumerate(lines):
        line = _line.strip()
        
        _a, _b = counter(line)
        posterior_a, posterior_b = prior_a + _a, prior_b + _b
        likelihood = binomial(_a , _b)
        
        print('case', i, ':', line)
        print('Likelihood:',likelihood)
        print('Beta prior:\ta =',prior_a,'b =', prior_b)
        print('Beta posterior: a =',posterior_a,'b =', posterior_b, end = '\n\n')
        
        prior_a = posterior_a
        prior_b = posterior_b

    fp.close()