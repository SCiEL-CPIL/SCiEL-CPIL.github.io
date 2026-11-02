import numpy as np
import scipy.optimize as opt

def _rank23_numerical():
    n = m = p = 3
    rank = 23
    
    # Target tensor T[i,j,k] = 1
    T_target = np.ones((n, m, p))
    
    def error(params):
        # params: flattened U, V, W
        # U: 9 x rank, V: 9 x rank, W: 9 x rank
        U = params[:n*m*rank].reshape((n*m, rank))
        V = params[n*m*rank:2*n*m*rank].reshape((m*p, rank))
        W = params[2*n*m*rank:].reshape((p*n, rank))
        
        U_r = U.reshape((n, m, rank))
        V_r = V.reshape((m, p, rank))
        W_r = W.reshape((p, n, rank))
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.sum((T_recon - T_target)**2)
    
    # Initialize
    # Use a heuristic initialization: start with rank-26 and add 0s for new cols?
    # Or random.
    # Let's try random initialization with small values.
    np.random.seed(42)
    x0 = np.random.randn(3 * n * m * rank) * 0.1
    
    # Optimize
    result = opt.minimize(error, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if result.fun < 1e-5:
        x = result.x
        U = x[:n*m*rank].reshape((n*m, rank))
        V = x[n*m*rank:2*n*m*rank].reshape((m*p, rank))
        W = x[2*n*m*rank:].reshape((p*n, rank))
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            
    return None

def construct():
    # ... existing code ...
    # Try numerical rank 23
    r23_num = _rank23_numerical()
    if r23_num:
        return r23_num
    # ... fallbacks ...
