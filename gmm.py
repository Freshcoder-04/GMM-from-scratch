import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k, n_iter=1000, cont_th=1e-6):
        self.k = k
        self.n_iter = n_iter
        self.cont_th = cont_th
        self.means = None
        self.covariances = None
        self.weights = None
        self.responsibilities = None

    def getParams(self):
        return {"means": self.means,"covariances": self.covariances,"weights": self.weights}

    def fit(self, X): # done
        n_samples, n_features = X.shape
        # Weights
        self.weights = np.random.rand(self.k)
        self.weights /= np.sum(self.weights)
        # Means
        self.means = []
        feature_min = np.min(X,axis=0)
        feature_max = np.max(X,axis=0)
        for i in range(self.k):
            self.means.append(feature_min+(feature_max-feature_min)*np.random.rand(n_features))
        self.means = np.array(self.means)

        # Covariances
        self.covariances = []
        for i in range(self.k):
            temp_mat = np.random.rand(n_features,n_features)*100
            temp_mat = np.dot(temp_mat,temp_mat.T)
            self.covariances.append(temp_mat)
        self.covariances = np.array(self.covariances)

        log_likelihood_old = None
        for iteration in range(self.n_iter):
            # Expectation step
            self.getMembership(X)

            # Maximization step
            self.MaximizationStep(X)
            log_likelihood_new = self.getLikelihood(X)
            if log_likelihood_old != None and abs(log_likelihood_new - log_likelihood_old) < self.cont_th:
                print(f"Converges after {iteration + 1} iterations")
                break
            log_likelihood_old = log_likelihood_new

    def getMembership(self,X): # done
        n_samples = X.shape[0]
        self.responsibilities = np.zeros((n_samples, self.k))
        for k in range(self.k):
            pdf = self.Gaussian(X, self.means[k], self.covariances[k])
            self.responsibilities[:, k] = self.weights[k] * pdf

        temp = self.responsibilities.sum(axis=1, keepdims=True)
        temp += 1e-6
        self.responsibilities /= temp
        return self.responsibilities

    def getLikelihood(self, X): # done
        log_likelihood = np.zeros((X.shape[0]))
        for k in range(self.k):
            pdf = self.Gaussian(X, self.means[k], self.covariances[k])
            log_likelihood += self.weights[k] * pdf
        return np.sum(np.log(log_likelihood))

    def MaximizationStep(self, X): # done
        n_samples,n_features = X.shape
        temp = self.responsibilities.sum(axis=0)
        self.weights = temp / n_samples
        covariances = np.zeros((self.k,n_features,n_features))
        self.means = np.dot(self.responsibilities.T, X) / temp[:, np.newaxis]
        for k in range(self.k):
            X_centered = X - self.means[k]
            covariances[k] = np.dot(self.responsibilities[:, k] * X_centered.T, X_centered) / temp[k]
        self.covariances = covariances

    def Gaussian(self, X, mean, covariance): # done
        n_features = X.shape[1]
        covariance += np.eye(n_features) * 1e-2
        pdf = multivariate_normal(mean=mean,cov=covariance).pdf(X)
        return pdf