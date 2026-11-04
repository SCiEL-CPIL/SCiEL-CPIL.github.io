import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Construct the target tensor T as a flat vector
    # T[a, b, c] where a in [0, nm), b in [0, mp), c in [0, pn)
    # T is 1 if a = i*m+j, b = j*p+k, c = k*n+i
    T_flat = np.zeros(nm * mp * pn, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = (a * mp + b) * pn + c
                T_flat[idx] = 1.0
    
    # Precompute indices for faster reconstruction
    # We need to map flat index to (a, b, c)
    # Or just iterate over all a, b, c
    
    # Function to compute residual given parameters x
    # x is flat vector of U, V, W
    # U is (nm, R), V is (mp, R), W is (pn, R)
    # x[0:R*nm] -> U
    # x[R*nm:R*nm+R*mp] -> V
    # x[R*nm+R*mp:R*nm+R*mp+R*pn] -> W
    
    def residual(x, R):
        U = x[:R*nm].reshape(R, nm).T  # (nm, R)
        V = x[R*nm:R*nm+R*mp].reshape(R, mp).T  # (mp, R)
        W = x[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(R, pn).T  # (pn, R)
        
        # Compute sum_r U_r \otimes V_r \otimes W_r
        # Efficiently:
        # M = np.einsum('ar,br,cr->abc', U, V, W)
        # This might be slow.
        # Alternative:
        # M = np.tensordot(U, V, axes=1) # (nm, mp, R)
        # M = np.einsum('abk,ck->abc', M, W)
        
        # Let's use einsum
        M = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        M_flat = M.flatten()
        return M_flat - T_flat

    def objective(x, R):
        r = residual(x, R)
        return np.dot(r, r)

    # Try ranks from 25 to 49
    # We start from lower ranks to find the minimum.
    # If we find a solution for R, we return it.
    # We must ensure we don't exceed time.
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Random seed for reproducibility in search, but different restarts
    np.random.seed(42)
    
    for R in range(25, 50):
        if time.time() > deadline:
            break
            
        found = False
        # Number of restarts depends on R and time left
        # For lower R, we might need more restarts
        max_restarts = 10
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            # Scale: values around 1
            x0 = np.random.randn(R * (nm + mp + pn))
            
            # Optimization
            try:
                res = opt.minimize(objective, x0, args=(R,), method='L-BFGS-B', 
                                   options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            except Exception as e:
                continue
                
            if res.success:
                r_val = res.fun
                if r_val < 1e-6:
                    # Try to round to integers or half-integers
                    x_opt = res.x
                    U_opt = x_opt[:R*nm].reshape(nm, R)
                    V_opt = x_opt[R*nm:R*nm+R*mp].reshape(mp, R)
                    W_opt = x_opt[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(pn, R)
                    
                    # Try rounding to integers
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    
                    # Verify integer solution
                    M_int = np.einsum('ar,br,cr->abc', U_int.T, V_int.T, W_int.T, optimize=True)
                    if np.allclose(M_int.flatten(), T_flat, atol=1e-9):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    
                    # Try rounding to half-integers (multiply by 2, round, divide by 2)
                    U_half = np.round(U_opt * 2) / 2.0
                    V_half = np.round(V_opt * 2) / 2.0
                    W_half = np.round(W_opt * 2) / 2.0
                    
                    M_half = np.einsum('ar,br,cr->abc', U_half.T, V_half.T, W_half.T, optimize=True)
                    if np.allclose(M_half.flatten(), T_flat, atol=1e-9):
                        return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                        
                    # If close but not exact, maybe we need to refine?
                    # Or maybe the solution is not "nice".
                    # But if the minimum rank is R, there should be a rational solution.
                    # If L-BFGS-B converges to a local minimum that is not the global one,
                    # we might miss it.
                    # However, for tensor rank, the landscape is complex.
                    
                    # If we found a very low error but couldn't round, 
                    # we might try to use the rounded values as a new start for a local search?
                    # Or just accept that this R might not work with this initialization.
                    
            # If not found, try next restart

    # If no lower rank found, return schoolbook
    R = nm * mp * pn # 50
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
