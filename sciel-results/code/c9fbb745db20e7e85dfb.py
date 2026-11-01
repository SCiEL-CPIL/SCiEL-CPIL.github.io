import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                T[a,b,c] = 1.0
    
    # Try rank 13
    for rank in [13, 12, 11]:
        best = None
        for trial in range(100):
            np.random.seed(trial)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            def loss(params):
                U_p = params[:n*m*rank].reshape(n*m, rank)
                V_p = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_p = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                T_recon = np.einsum('ad,bd,cd->abc', U_p, V_p, W_p)
                return np.sum((T_recon - T)**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 1000})
            
            if res.fun < 1e-10:
                U_p = res.x[:n*m*rank].reshape(n*m, rank)
                V_p = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_p = res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                best = (U_p, V_p, W_p)
                break
        
        if best is not None:
            U, V, W = best
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to rank 14 (Strassen)
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    # ... (same as current)
