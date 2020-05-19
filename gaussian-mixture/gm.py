
def gmm(X, n_classes, n_iter):
    # TODO fill in your code here
    #X is NxK, each row X[i] is a data point

    N = X.shape[0]
    D = X.shape[1]
    K = n_classes


    global pi, mu, cov

    pi = np.array([1/K for _ in range(K)]) #?
    cov = np.array([[[random.random()  for k in range(D)] for k_ in range(D)] for _ in range(K)]) #?
    mu = np.array([[random.random() for k in range(D)] for _ in range(K)])
    mu = np.array([m/sum(mu) for m in mu])

    assert(pi.shape == (K, ))
    assert(cov.shape == (K, D, D))
    assert(mu.shape == (K, D))

    #assert(mu.shape == (K,))

    def gaussian(x, mu, cov):
        #x is 1xK (1x4), cov is dxd for this gaussian
        #There are k (3) gaussians
        #returns probability that gaussian mean, cov generated it
        lhs = 1 / ((2* np.pi) * abs(np.linalg.det(cov))**(1./2.))
        rhs = np.exp(- (1/2) * np.dot(np.dot((x - mu).T, np.linalg.inv(cov)), (x - mu)))

        return lhs* rhs 

    def gamma(X, mean, cov, pi, k_):
        #return probability for gaussian over k partitions
        ans = [pi[k_] * gaussian(X[i], mean, cov) for i in range(N)] / \
               sum([sum([pi[k]*gaussian(X[i_], mean, cov) for i_ in range(N)]) for k in range(K)])

        return np.array(ans) #[pi[k_] * gaussian(x, mean, cov) for k in range(K)] / sum([pi[k]*gaussian(x, mean, cov) for k in range(K)])

    def W(i, j):
        return pi[j] * gaussian(x[i], mu[j], cov[j])

    def est_params(pi, mu, cov):

        #W = [pi[i] * gaussian(i) for i in range(K)]
        #W = [w/sum(W) for w in W]

        pi = [np.dot(X[:, k], gamma(X, mu[k], cov[k], pi, k)) for k in range(K)]
        assert(np.array(pi).shape == (K, ))
        #sum to 1.
        pi = np.array([p/sum(pi) for p in pi])

        print(1)
        #mu = [sum([np.dot(X[n], gamma(n, k)) for n in range(N)])  / 
        #             sum([gamma(n_, k) for n_ in range(N)]) for k in range(K)]

        mu = [np.dot(X.T, gamma(X, mu[k], cov[k], pi, k)) for k in range(K)]
        mu = np.array(mu)
        #mu = [np.dot(np.array([X[n] for n in range(N)]), np.array([gamma(X[n], mu, cov, pi) for n in range(N)]))]# for k in range(K)]
        #mu = np.array([m/sum(mu) for m in mu])
        print(2)
        #assert mu.shape == (4,)
        print(X.shape)
        print(np.array(mu).shape)

        #cov = [pi[k]* sum([gamma(n, k) * np.matmul((X[n] - mu), (X[n] - mu).T) for n in range(N)]) / 
        #       sum([gamma(n_, k) for n_ in range(N)])
        #       for k in range(K)]
        for k in range(K):
            x_norm = X - mu[k, :]
            assert(np.array(x_norm).shape == (N, D))

            t2 = np.matrix(np.diag(gamma(X, mu[k], cov[k], pi, k)))
            t1 = X[:, :] - mu[k, :]
            t3 = (X[:, :] - mu[k, :])

            print(np.array(t1).shape, np.array(t2).shape, np.array(t3).shape)
            a = t1.T * t2
            b = a * t3
            print(np.array(b).shape)
            new = pi[k] * b
            cov[k, :, :] = new
            #cov[k, :, :] = [pi[k]* sum([gamma(X, mu[k], cov[k], pi, k) * np.matmul((X[:, k] - mu[k]), (X[:, k] - mu[k]).T) ])]
        #cov = np.array([e/sum(cov) for e in cov])

        #assert(len(pi) == K)
        #assert(len(mu) == K)
        #assert(len(cov) == K and len(cov[0]) == K)

        return pi, mu, cov

    for _ in range(2):
        print("p", pi, "m", mu, "e", cov)
        print("")
        pi, mu, cov = est_params(pi, mu, cov)


    print(mu, cov)
    class_ = [np.argmax([gamma(X[i, k]) for k in range(K)] for i in range(N))]
    return np.array(class_, mu, cov)
    #assignments, mean, cov