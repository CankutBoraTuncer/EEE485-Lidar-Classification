import numpy as np
from scipy import optimize


class KernelSvmClassifier:
    def __init__(self, kernel):
        self.kernel = kernel
        self.alpha = None
        self.supportVectors = None
        self.supportAlphaY = None
    
    def fit(self, X, y):
        N = len(y)
        hXX = np.apply_along_axis(lambda x1: np.apply_along_axis(lambda x2: self.kernel(x1, x2), 1, X), 1, X)
        yp = y.reshape(-1, 1)
        GramHXy = hXX * np.matmul(yp, yp.T)

        # Dual problem functions
        def Ld0(G, alpha):
            return alpha.sum() - 0.5 * alpha.dot(alpha.dot(G))
        def Ld0dAlpha(G, alpha):
            return np.ones_like(alpha) - alpha.dot(G)

        # Constraints
        A = np.vstack((-np.eye(N), np.eye(N)))
        b = np.hstack((np.zeros(N),  np.ones(N)))
        constraints = ({'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y},
                       {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A})

        # Minimization
        optRes = optimize.minimize(fun=lambda a: -Ld0(GramHXy, a),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda a: -Ld0dAlpha(GramHXy, a), 
                                   constraints=constraints)
        self.alpha = optRes.x

        epsilon = 1e-8
        supportIndices = self.alpha > epsilon
        self.supportVectors = X[supportIndices]
        self.supportAlphaY = y[supportIndices] * self.alpha[supportIndices]

    def predict(self, X):
        def predict1(x):
            x1 = np.apply_along_axis(lambda s: self.kernel(s, x), 1, self.supportVectors)
            x2 = x1 * self.supportAlphaY
            return np.sum(x2)
        d = np.apply_along_axis(predict1, 1, X)
        return 2 * (d > 0) - 1

class MultiClassSVM:
    def __init__(self, kernel):
        self.classifiers = {}
        self.kernel = kernel
        self.svm = None
    
    def train(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            self.svm = KernelSvmClassifier(self.kernel)
            self.svm.fit(X, y_binary)
            self.classifiers[cls] = self.svm

    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers.values()]).T
        return self.classes_[np.argmax(predictions, axis=1)]
