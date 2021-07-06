import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from Scripts.plotka import plot_decision_regions
from Scripts.reglog import LogisticRegressionGD, MultiLogisticRegressionGD


class Perceptron(object):

    def __init__(self, eta=0.01, n=10):
        self.eta = eta
        self.n = n

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        # self.errors_ = []

        for _ in range(self.n):
            # errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                # errors += int(update != 0.0)
            # self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class MultiPerceptron:
    def __init__(self, pn1, pn2):
        self.pn1 = pn1
        self.pn2 = pn2

    def predict(self, X):
        return np.where(self.pn1.predict(X) == 1, 0,
                        np.where(self.pn2.predict(X) == 1, 2, 1))


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # train1
    X_train_01_subset = np.copy(y_train)
    X_train_01_subset = X_train_01_subset[(X_train_01_subset != 2)]
    y_train_01_subset = np.copy(X_train)
    y_train_01_subset = y_train_01_subset[(y_train != 2)]

    X_train_01_subset[(X_train_01_subset != 0)] = -1
    X_train_01_subset[(X_train_01_subset == 0)] = 1

    pn1 = Perceptron(eta=0.1, n=10)
    pn1.fit(y_train_01_subset, X_train_01_subset)

    # train2
    X_train_02_subset = np.copy(y_train)
    X_train_02_subset = X_train_02_subset[(X_train_02_subset != 0)]
    y_train_02_subset = np.copy(X_train)
    y_train_02_subset = y_train_02_subset[(y_train != 0)]

    X_train_02_subset[(X_train_02_subset != 2)] = -1
    X_train_02_subset[(X_train_02_subset == 2)] = 1

    pn2 = Perceptron(eta=0.1, n = 1000)
    pn2.fit(y_train_02_subset, X_train_02_subset)

    mpn = MultiPerceptron(pn1, pn2)

    plot_decision_regions(X=X_test, y=y_test, classifier=mpn)
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')
    plt.legend(loc='upper left')
    plt.show()

    # Logistic regresions
    X_train_01_subset[(X_train_01_subset != 1)] = 0
    reglog1 = LogisticRegressionGD(eta=0.05, n=1000, random_state=1)
    reglog1.fit(y_train_01_subset, X_train_01_subset)

    X_train_02_subset[(X_train_02_subset != 1)] = 0
    reglog2 = LogisticRegressionGD(eta=0.05, n=2000, random_state=1)
    reglog2.fit(y_train_02_subset, X_train_02_subset)
    # 1st regresion
    probability1 = reglog1.probability(y_train_01_subset)
    print("1st regresion")
    for i in probability1:
        print('%1.4f' % i, end="\n")
    # 2nd regresion
    probability2 = reglog2.probability(y_train_02_subset)
    print("2nd regresion")
    for i in probability2:
        print('%1.4f' % i, end="\n")

    multiReglog = MultiLogisticRegressionGD(reglog1, reglog2)

    plot_decision_regions(X=X_test, y=y_test, classifier=multiReglog)
    plt.title('Logistic regresion')
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
