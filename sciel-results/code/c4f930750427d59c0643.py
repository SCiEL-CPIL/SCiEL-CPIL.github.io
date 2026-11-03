import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = 3, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    def unpack(x, r):
        U = x[:nm*r].reshape(nm, r)
        V = x[nm*r:nm*r+mp*r].reshape(mp, r)
        W = x[nm*r+mp*r:].reshape(pn, r)
        return U, V, W
    
    def make_residuals(r):
        def residuals(x):
            U, V, W = unpack(x, r)
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            res = result - T
            return res.flatten()
        return residuals
    
    def try_rank(r, max_nfev=5000):
        if time.time() > deadline:
            return None
        
        total_vars = nm * r + mp * r + pn * r
        residuals_fn = make_residuals(r)
        
        # Try multiple initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            if attempt == 0:
                # Structured initialization based on schoolbook
                x0 = np.zeros(total_vars)
                # Set up some basic structure
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i*m+j < nm:
                                x0[i*m+j * r + min(i*m+j, r-1)] = 1
                            if j*p+k < mp:
                                x0[nm*r + (j*p+k) * r + min(j*p+k, r-1)] = 1
                            if k*n+i < pn:
                                x0[nm*r+mp*r + (k*n+i) * r + min(k*n+i, r-1)] = 1
            elif attempt == 1:
                # Random small values
                x0 = np.random.randn(total_vars) * 0.5
            else:
                # Random with larger scale
                x0 = np.random.randn(total_vars)
            
            try:
                sol = least_squares(residuals_fn, x0, method='trf', max_nfev=max_nfev)
                
                if sol.cost < 1e-8:
                    U, V, W = unpack(sol.x, r)
                    # Round to nearest integer or half-integer
                    U2 = np.round(U * 2) / 2
                    V2 = np.round(V * 2) / 2
                    W2 = np.round(W * 2) / 2
                    
                    # Verify
                    U3 = U2.reshape(n, m, r)
                    V3 = V2.reshape(m, p, r)
                    W3 = W2.reshape(p, n, r)
                    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                    
                    if np.allclose(result, T, atol=1e-9):
                        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                        
            except Exception as e:
                continue
        
        return None
    
    # Try ranks from low to high
    # Lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But we can try below that too since it's a lower bound, not guaranteed tight
    for r in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
