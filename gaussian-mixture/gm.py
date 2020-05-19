
def gmm(X, n_classes, n_iter):
    #X is NxK, each row X[i] is a data point

    N = X.shape[0]
    K = X.shape[1]
    print("N", N, "K", K)

    global pi, mu, eps
    pi = np.array([1/K for _ in range(K)]) #?
    size_eps = 1 #?
    eps = np.array([1/K for _ in range(K)]) #?
    mu = np.array([[1/K for _ in range(K)] for _ in range(K)])

    def gaussian(i):
        lhs = 1 / ((2* pi[i])**K * K**(1/2))
        rhs = np.exp(- (1/2) * (X[i]  - mu[i]).T * eps[i]**(-1) * (X[i] - mu[i]))

        return lhs* rhs 

    def gamma(n, k):
        return pi[k] * gaussian(k) / sum([pi[j]*gaussian(j) for j in range(K)])

    def est_params(pi, mu, eps):
        pi = [sum([gamma(j, k) for j in range(N)])/N for k in range(K)]

        mu = [sum([np.dot(X[n], gamma(n, k)) for n in range(N)])  / 
                     sum([gamma(n_, k) for n_ in range(N)]) for k in range(K)]

        eps = [pi[k]* sum([gamma(n, k) * np.matmul((X[n] - mu), (X[n] - mu).T) for n in range(N)]) / 
               sum([gamma(n_, k) for n_ in range(N)])
               for k in range(K)]

        return pi, mu, eps

    for _ in range(10):
        print(pi, mu, eps)
        pi, mu, eps = est_params(pi, mu, eps)


    print(mu, eps)
    return np.array([0 for _ in range(N)]), mu, eps

