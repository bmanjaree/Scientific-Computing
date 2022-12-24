import numpy as np
def cheb(N):
    # N is the number of points in the interior.
    if N==0:
        D = 0
        x = 1
        return D, x
    vals = np.linspace(0, N, N+1)
    x = np.cos(np.pi*vals/N)
    x = x.reshape(-1, 1)
    c = np.ones(N-1)
    c = np.pad(c, (1,), constant_values = 2)
    c *= (-1)**vals
    c = c.reshape(-1, 1)
    X = np.tile(x, (1, N+1))
    dX = X-X.T                  
    D  = (c*(1/c.T))/(dX+(np.eye(N+1)))       #off-diagonal entries
    D  = D - np.diag(sum(D.T))                #diagonal entries

    return D, x
