#############
##Problem 2##
#############

random.seed(1234)

###2.1###

def DGP(beta=np.array([0.3]*5 + [0]*95),N=1000,p=100,rho=0.2):
    u = scipy.stats.logistic.rvs(size=(N,1)) #errors
    cov = np.ones((p,p))*rho - np.eye(p,p)*rho + np.eye(p,p) #var-cov matrix
    X = np.random.multivariate_normal(mean=np.zeros(p),cov=cov,size=N) #X matrix
    y_1 = X@beta.reshape(-1,1) + u #list of y's
    Y = np.where(y_1 > 0, 1, 0) #make y's binary
    return (X, Y)
        
        
        
###2.2###

#now we do MLE


def LLLH(params): #Need to make this way more efficient!!!
    x, y = DGP()
    items = []
    for i in range(1000):
        component = -1 * np.log(1 + np.exp(x[i,:]@params)) + y[i]*(x[i,:]@params)
        items.append(component)
    logllh = np.sum(items)
    return -logllh

results = []
M=5000

for i in range(M):
    res = scipy.optimize.minimize(LLLH, np.array([0]*100), method='L-BFGS-B')
    results.append(res.x)

np.mean(results, axis=0)[0:5]
