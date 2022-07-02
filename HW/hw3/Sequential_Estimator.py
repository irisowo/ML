import argparse
from DataGenerator import Data_Generator_Univar

def get_parser():
    parser = argparse.ArgumentParser(description='input m s')
    parser.add_argument('-m', type = float, default=3.0)
    parser.add_argument('-s', type = float, default=5.0)
    args = parser.parse_args()
    return args


# ref : https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
def update(ex_Aggregate, new_Value):
    (cnt, m, M2, var) = ex_Aggregate
    cnt += 1
    delta = new_Value - m # xn - m(n-1)
    m += delta / cnt
    delta2 = new_Value - m # xn - m(n)
    M2 += delta * delta2
    return (cnt, m, M2, M2/cnt)


def online_algo(m, s) :
    # Initialize
    new_point =  Data_Generator_Univar(m, s)
    ex_Aggregate = (1, new_point, new_point, 0.0) # (cnt, mean, M2, var)
    print("Add data point: ",new_point)
    print("Mean: ",new_point, " Variance: ", 0.0)
    
    m_err, var_err = 1.0, 1.0
    while(m_err > 0.001 or var_err > 0.001) :
        new_point =  Data_Generator_Univar(m, s)
        print("Add data point: ",new_point)
        
        new_Aggregate = update(ex_Aggregate, new_point)
        print("Mean: ",new_Aggregate[1], " Variance: ", new_Aggregate[3])
        
        m_err = abs(new_Aggregate[1] - ex_Aggregate[1])
        var_err = abs(new_Aggregate[3] - ex_Aggregate[3])
        ex_Aggregate = new_Aggregate


if __name__ == '__main__':
    args = get_parser()
    m, s = args.m, args.s
    print("Data point source function: N({}, {})\n\n".format(m, s))
    online_algo(m, s)
