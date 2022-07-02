import numpy as np
import matplotlib.pyplot as plt


def design_matrix(x, n):
    X = np.array([ (x ** i) for i in range(n)])
    return X.reshape((1, n))


# class BayesianLinearRegression:
#   def print_post_mean_var(self)
#   def visualization(self, num_point=50)
def print_post_mean_var(self):
    n = self.n
    print("Postirior mean:")
    for i in range(n):
        print("{:15.10f}".format(self.post_mean[i, 0]))
    print()

    print("Postirior variance:")
    for i in range(n):
        for j in range(n):
            if j < n-1:
                print("{:15.10f}".format(self.post_var[i, j]), end=', ')
            else:
                print("{:15.10f}".format(self.post_var[i, j]))
    print()


def visualization(self, num_point=100):
    n, w, a = self.n, self.w, self.a # a
    data_x, data_y = self.x, self.y
    data_mean, data_var = self.mean, self.var

    x = np.linspace(-2.0, 2.0, num_point)
    X = []
    for i in range(num_point):
        X.append(design_matrix(x[i], n))

    # 1 : Ground truth
    plt.subplot(221)
    plt.title("Ground truth")
    func = np.poly1d(np.flip(w))
    y = func(x)
    draw(x, y, var=a)

    # 2 : Predict result
    plt.subplot(222)
    plt.title("Predict result")
    func = np.poly1d(np.flip(np.reshape(data_mean[2], n)))
    y = func(x)
    var = np.zeros((num_point))
    for i in range(num_point):
        var[i] = a + X[i].dot(data_var[2].dot(X[i].T))[0][0]
    plt.scatter(data_x, data_y, s=10, alpha=0.5)
    draw(x, y, var)

    # 3 : iter = 10
    plt.subplot(223)
    plt.title("After 10 incomes")
    func = np.poly1d(np.flip(np.reshape(data_mean[0], n)))
    y = func(x)
    for i in range(num_point):
        var[i] = a + (X[i] @ data_var[0] @ X[i].T)[0][0]
    plt.scatter(data_x[:10], data_y[:10], s=7.0, alpha=0.5)
    draw(x, y, var)

    # 4 : iter = 50
    plt.subplot(224)
    plt.title("After 50 incomes")
    func = np.poly1d(np.flip(np.reshape(data_mean[1], n)))
    y = func(x)
    for i in range(num_point):
        var[i] = a + (X[i] @ data_var[1] @ X[i].T)[0][0]
    plt.scatter(data_x[:50], data_y[:50], s=7.0, alpha=0.5)
    draw(x, y, var)

    plt.tight_layout()
    plt.show()


def draw(x, y, var):
	plt.plot(x, y, color = "k")
	plt.plot(x, y + var, color = "r")
	plt.plot(x, y - var, color = "r")
	plt.xlim(-2.0, 2.0)
	plt.ylim(-20.0, 25.0)
