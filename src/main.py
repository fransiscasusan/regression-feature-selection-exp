import numpy as np
import pandas as pd
from timeit import default_timer as timer
from sklearn import linear_model
import sklearn
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import argparse

'''
Generating the correlated data matrix X with dimension n x p, where n is the number of data
and p is the dimension of each data point; rho is the correlation between two different coordinates;
'''
def generateX(n,rho,p):
    cov_matrix = rho*np.ones((p,p))
    cov_matrix = cov_matrix + (1-rho)*np.identity(p)
    X = np.random.multivariate_normal(np.zeros(p), cov_matrix, n)
    return(X)

'''
Generating beta = (1,1,...,1,0,...,0), where only k coordinates are non-zero
'''
def generateStrongSignal(p,k):
    beta = np.zeros(p)
    for i in range(k):
        beta[i] = 1.0
    return(beta)

'''
Generating beta = (0.1,0.1,...,0.1,0,...,0), where only k coordinates are non-zero
'''
def generateWeakSignal(p,k):
    beta = np.zeros(p)
    for i in range(k):
        beta[i] = 1.0
    return(0.01*beta)

'''
Generating beta = (1/k,2/k,...,1,0,...,0), where only k coordinates are non-zero
'''
def generateVariedSignal(p,k):
    beta = np.zeros(p)
    for i in range(k):
        beta[i] = (i+1)*1.0/k
    return(beta)

'''
Generating beta based on type of signal, where type_of_signal: 1. weak, 2. strong, 3. varied
'''
def generateBeta(p,k,type_of_signal):
    if type_of_signal not in [1,2,3]:
        print("Signal Error")
        return(None)
    if type_of_signal == 1:
        return(generateWeakSignal(p,k))
    elif type_of_signal == 2:
        return(generateStrongSignal(p,k))
    else:
        return(generateVariedSignal(p,k))

'''
Generating y, given the true beta value and data X, with some Gaussian noise epsilon
'''
def generateY(X,true_beta):
    n,p = X.shape
    epsilon = np.random.multivariate_normal(np.zeros(n), np.identity(n))
    y = X.dot(true_beta) + epsilon
    return(y)

'''
====================================================================================================================
                                        MODEL IMPLEMENTATION
====================================================================================================================
'''

'''
Orthogonal Matching Pursuit ALGORITHM
'''
def run_omp(X,y,true_beta,k):
    start = timer()

    n, p = X.shape    
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
    omp.fit(X, y)
    beta = omp.coef_
    
    ypred = omp.predict(X)    
    squared_error_sum = np.sum(np.square(ypred-y))
    
    end = timer()
    Time = end-start
    
    RMSE = np.sqrt(squared_error_sum*1.0/n)
    DR = (np.sum((np.multiply(true_beta,beta)!=0)*1.0))*1.0/k
    
    SS_tot = np.sum(np.square(y-np.mean(y)))
    Rsquared = 1-squared_error_sum*1.0/SS_tot

    print(RMSE,DR,Rsquared,Time,'OMP')    
    return(RMSE,DR,Rsquared,Time,squared_error_sum*1.0/n)

'''
LASSO ALGORITHM
'''
def run_Lasso(X,y,true_beta,k,CURRENT_RANDOM_SEED=1):
    start = timer()
    n, p = X.shape
    
    lasso_model = linear_model.Lasso(alpha = 1.0,random_state=CURRENT_RANDOM_SEED)
    lasso_model.fit(X,y)
    beta = lasso_model.coef_
    ypred = lasso_model.predict(X)
    
    squared_error_sum = np.sum(np.square(ypred-y))
    
    end = timer()
    Time = end-start
    
    RMSE = np.sqrt(squared_error_sum*1.0/n)
    DR = (np.sum((np.multiply(true_beta,beta)!=0)*1.0))*1.0/k
    
    SS_tot = np.sum(np.square(y-np.mean(y)))
    Rsquared = 1-squared_error_sum*1.0/SS_tot
    
    print(RMSE,DR,Rsquared,Time,'LASSO')    
    return()

'''
Truncation function for TSGD Algorithm
'''
def truncate(beta,k):
    p = beta.shape[0]
    sorted_indices = np.abs(beta).argsort()[::-1].tolist()
    dummy = np.zeros(p)
    for el in sorted_indices[:k]:
        dummy[el] = 1.0
    return(np.multiply(beta,dummy))

'''
TSGD ALGORITHM
'''
def run_TSGD(X,y,true_beta,k,min_offline_errors):
    start = timer()

    n, p = X.shape
    ns = NS
    ns = [nn-1 for nn in ns]
    n_idx = 0
    eta = np.log(ns[n_idx])*1.0/ns[n_idx]

    squared_error_sum = 0
    true_support_sum = 0

    beta = np.random.rand(p)-0.5
    for i in range(n):
        prev_beta = beta
        y_pred = X[i,:].dot(prev_beta)
        
        loss_i = (y_pred-y[i,])**2
        current_detection_rate = (np.sum((np.multiply(true_beta,prev_beta)!=0)*1.0))*1.0/k
        squared_error_sum += loss_i
        true_support_sum += current_detection_rate
        
        beta = prev_beta + eta*(y[i]-np.dot(X[i,:],prev_beta))*X[i,:]
        beta = truncate(beta,k)

        if i in ns:
            end = timer()
            Time = end-start
            
            RMSE = np.sqrt(squared_error_sum*1.0/i)
            DR = true_support_sum*1.0/i

            SS_tot = np.sum(np.square(y[:i+1]-np.mean(y[:i+1])))
            Rsquared = 1-squared_error_sum*1.0/SS_tot
            
            regret = squared_error_sum*1.0/i - min_offline_errors[n_idx]

            if n_idx != len(ns) - 1:
                n_idx += 1
                eta = np.log(ns[n_idx])*1.0/ns[n_idx]

            print(RMSE,DR,Rsquared,Time,regret,i+1,'TSGD')
    
    return()

'''
OLST ALGORITHM
'''
def run_OLST(X,y,true_beta,k,rho,min_offline_errors):
    start = timer()
    n, p = X.shape
    ns = NS
    ns = [nn-1 for nn in ns]
    n_idx = 0
    y = y.reshape((n,1))

    squared_error_sum = 0
    true_support_sum = 0

    beta = np.random.rand(p)-0.5

    #INITIALIZATION
    mu_x = np.zeros((p))
    mu_y = 0
    S_xx = np.zeros((p,p))
    S_xy = np.zeros((p,1))

    for i in range(n):

        prev_beta = beta
        y_pred = X[i,:].dot(prev_beta)
        loss_i = (y_pred-y[i,0])**2
        current_detection_rate = (np.sum((np.multiply(true_beta,prev_beta)!=0)*1.0))*1.0/k
        
        
        squared_error_sum += loss_i
        true_support_sum += current_detection_rate
                    
        # Finding $\hat{beta}$ by OLS
        X_current, y_current = X[i,:].reshape((1,p)), y[i]
        mu_x = i*mu_x/(i+1) + X_current*1.0/(i+1)
        mu_y = i*mu_y/(i+1) + y_current*1.0/(i+1)
        S_xx = S_xx*i/(i+1) + np.dot(X_current.T,X_current)*1.0/(i+1)
        S_xy = S_xy*i/(i+1) + y_current*X_current.T*1.0/(i+1)
    
        # Normalization
        if i>=1:
            Pi = np.linalg.inv(np.sqrt(np.multiply((S_xx - np.square(mu_x)),np.identity(p))))
            S_xx_n = np.dot(Pi, S_xx-np.dot(mu_x,mu_x.T))
            S_xx_n = np.dot(S_xx_n,Pi)

            S_xy_n = np.dot(Pi, S_xy) - mu_y*np.dot(Pi, mu_x.T).reshape((p,1))

        clf = Ridge(alpha=0.001)
        if k==10:
            clf = Ridge(alpha=0.001)
        if k==100 and rho!=0:
            clf = Ridge(alpha=1.0)
        
        if i==0:
            clf.fit(S_xx,S_xy)
        else:
            clf.fit(S_xx_n,S_xy_n)
        beta = clf.coef_.reshape((p,))

        # Keeping only k variables with largest $|\hat{\beta}_j|$
        sorted_indices = np.abs(beta).argsort()[::-1].tolist()
        k_biggest_indices = np.sort(sorted_indices[:k])
        
        # Fitting the model on the selected features by OLS
        selected_X = X[:i+1,k_biggest_indices]

        clf = Ridge(alpha=0.0001)
        if k==10:
            clf = Ridge(alpha=0.0001)
        if k==100 and rho!=0:       
            clf = Ridge(alpha=0.1)
        clf.fit(selected_X, y[:i+1])
        OLS_beta = clf.coef_.reshape((k,))
        
#         OLS_beta = np.dot(np.linalg.inv(np.dot(selected_X.T,selected_X)),np.dot(selected_X.T,y_current))
                
        beta = np.zeros(p)
        for j,ind in enumerate(k_biggest_indices):
            beta[ind] = OLS_beta[j]

        if i in ns:
            end = timer()
            Time = end-start
            
            RMSE = np.sqrt(squared_error_sum*1.0/i)
            DR = true_support_sum*1.0/i

            SS_tot = np.sum(np.square(y[:i+1]-np.mean(y[:i+1])))
            Rsquared = 1-squared_error_sum*1.0/SS_tot

            regret = squared_error_sum*1.0/n - min_offline_errors[n_idx]

            if n_idx != len(ns) - 1:
                n_idx += 1

            print(RMSE,DR,Rsquared,Time,regret,i+1,'OLST')
    
    return()

'''
Hybrid ALGORITHM
'''
def run_hybrid(X,y,true_beta,k,rho,min_offline_errors,BURNIN):
    start = timer()
    n, p = X.shape
    ns = NS
    ns = [nn-1 for nn in ns]
    n_idx = 0
    eta = np.log(ns[n_idx])*1.0/ns[n_idx]
    y = y.reshape((n,1))

    squared_error_sum = 0
    true_support_sum = 0

    beta = np.random.rand(p)-0.5

    #INITIALIZATION
    mu_x = np.zeros(p)
    mu_y = 0
    S_xx = np.zeros((p,p))
    S_xy = np.zeros((p,1))

    for i in range(BURNIN):

        prev_beta = beta
        y_pred = X[i,:].dot(prev_beta)
        loss_i = (y_pred-y[i,0])**2
        current_detection_rate = (np.sum((np.multiply(true_beta,prev_beta)!=0)*1.0))*1.0/k
        
        squared_error_sum += loss_i
        true_support_sum += current_detection_rate
        
        # Finding $\hat{beta}$ by OLS
        X_current, y_current = X[i,:].reshape((1,p)), y[i]
        mu_x = i*mu_x/(i+1) + X_current*1.0/(i+1)
        mu_y = i*mu_y/(i+1) + y_current*1.0/(i+1)
        S_xx = S_xx*i/(i+1) + np.dot(X_current.T,X_current)*1.0/(i+1)
        S_xy = S_xy*i/(i+1) + y_current*X_current.T*1.0/(i+1)
    
        # Normalization
        if i>=1:
            Pi = np.linalg.inv(np.sqrt(np.multiply((S_xx - np.square(mu_x)),np.identity(p))))
            S_xx_n = np.dot(Pi, S_xx-np.dot(mu_x,mu_x.T))
            S_xx_n = np.dot(S_xx_n,Pi)

            S_xy_n = np.dot(Pi, S_xy) - mu_y*np.dot(Pi, mu_x.T).reshape((p,1))


        clf = Ridge(alpha=0.001)
        if k==10:
            clf = Ridge(alpha=0.001)
        if k==100 and rho!=0:
            clf = Ridge(alpha=1.0)
        
        if i==0:
            clf.fit(S_xx,S_xy)
        else:
            clf.fit(S_xx_n, S_xy_n)
        beta = clf.coef_.reshape((p,))

        # Keeping only k variables with largest $|\hat{\beta}_j|$
        sorted_indices = np.abs(beta).argsort()[::-1].tolist()
        k_biggest_indices = np.sort(sorted_indices[:k])
        
        # Fitting the model on the selected features by OLS
        selected_X = X[:i+1,k_biggest_indices]

        clf = Ridge(alpha=0.0001)
        if k==10:
            clf = Ridge(alpha=0.0001)
        if k==100 and rho!=0:       
            clf = Ridge(alpha=0.1)
        clf.fit(selected_X, y[:i+1])
        OLS_beta = clf.coef_.reshape((k,))
        
#         OLS_beta = np.dot(np.linalg.inv(np.dot(selected_X.T,selected_X)),np.dot(selected_X.T,y_current))
                
        beta = np.zeros(p)
        for j,ind in enumerate(k_biggest_indices):
            beta[ind] = OLS_beta[j]

        if i in ns:
            end = timer()
            Time = end-start
            
            RMSE = np.sqrt(squared_error_sum*1.0/i)
            DR = true_support_sum*1.0/i

            SS_tot = np.sum(np.square(y[:i+1]-np.mean(y[:i+1])))
            Rsquared = 1-squared_error_sum*1.0/SS_tot

            regret = squared_error_sum*1.0/n - min_offline_errors[n_idx]

            if n_idx != len(ns) - 1:
                n_idx += 1
                eta = np.log(ns[n_idx])*1.0/ns[n_idx]

            print(RMSE,DR,Rsquared,Time,regret,i+1,'OLST-hybrid')

    y = y.reshape((n,))
    for i in range(BURNIN, n):

        prev_beta = beta
        y_pred = X[i,:].dot(prev_beta)
        
        loss_i = (y_pred-y[i,])**2
        current_detection_rate = (np.sum((np.multiply(true_beta,prev_beta)!=0)*1.0))*1.0/k
        squared_error_sum += loss_i
        true_support_sum += current_detection_rate
        
        beta = prev_beta + eta*(y[i]-np.dot(X[i,:],prev_beta))*X[i,:]
        beta = truncate(beta,k)

        if i in ns:
            end = timer()
            Time = end-start
            
            RMSE = np.sqrt(squared_error_sum*1.0/i)
            DR = true_support_sum*1.0/i

            SS_tot = np.sum(np.square(y[:i+1]-np.mean(y[:i+1])))
            Rsquared = 1-squared_error_sum*1.0/SS_tot
            
            regret = squared_error_sum*1.0/i - min_offline_errors[n_idx]

            if n_idx != len(ns) - 1:
                n_idx += 1
                eta = np.log(ns[n_idx])*1.0/ns[n_idx]

            print(RMSE,DR,Rsquared,Time,regret,i+1,'TSGD-hybrid')
    
    return()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('k', type=int, help='number of important features')
    parser.add_argument('rho', type=float, help='correlation between coordinates')
    parser.add_argument('sig', type=int, help='the strength of beta params (1 for weak, 2 for strong, 3 for varied)')
    parser.add_argument('model', type=int, help='1:Lasso, 2:TSGD, 3:OLST, 4:hybrid')
    parser.add_argument('--BURNIN', type=int, help='For hybrid, when to switch between TSGD and OLST')

    args = parser.parse_args()

    k               = args.k
    rho             = args.rho
    sig             = args.sig
    model_num       = args.model
    num_batch       = 10

    true_beta     = generateBeta(1000,k,sig)
    Xs = []
    ys = []
    for i in range(num_batch):
        Xi = generateX(1000,rho,1000)
        Xs.append(Xi)
        ys.append(generateY(Xi,true_beta).reshape((1000,1)))
    X = np.vstack(Xs)
    y = np.vstack(ys)


    NS              = [1000,5000,10000]
    # true_beta = generateBeta(100,10,2)
    # X = generateX(1000,rho,100)
    # print("shape",X.shape)
    # y = generateY(X, true_beta)
    # NS = [500,1000]

    if model_num == 1:
        for n in NS:
            Xn          = X[:n,:]
            yn          = y[:n]
            run_Lasso(Xn,yn,true_beta,k)

    else:
        min_offline_errors = []
        for n in NS:
            Xn        = X[:n,:]
            yn         = y[:n]
            res_omp = run_omp(Xn,yn,true_beta,k)
            min_offline_errors.append(res_omp[-1])

        if model_num == 2:
            run_TSGD(X,y,true_beta,k,min_offline_errors)

        if model_num == 3:
            run_OLST(X,y,true_beta,k,rho,min_offline_errors)

        if model_num == 4:
            BURNIN     = args.BURNIN
            run_hybrid(X,y,true_beta,k,rho,min_offline_errors,BURNIN)
