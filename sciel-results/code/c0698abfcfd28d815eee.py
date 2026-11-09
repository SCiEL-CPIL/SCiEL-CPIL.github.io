import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_sb

    # Try to find lower rank
    # We'll try ranks from best_rank - 1 down to some lower bound
    # Lower bound is max(n*m, m*p, p*n) = 21
    # Let's try down to 35 to be safe with time
    
    for r in range(best_rank - 1, 34, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple random restarts
        # Number of restarts depends on remaining time
        # Each least_squares call can take a few seconds
        # Let's allow ~5 restarts per rank if time permits
        
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W
            # Random initialization
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(x):
                U_flat = x[:n*m*r]
                V_flat = x[n*m*r : n*m*r + m*p*r]
                W_flat = x[n*m*r + m*p*r :]
                
                U_m = U_flat.reshape(n*m, r)
                V_m = V_flat.reshape(m*p, r)
                W_m = W_flat.reshape(p*n, r)
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Sum over c
                            # U[i*m+j, :] dot (V[j*p+k, :] * W[k*n+i, :])
                            val = np.dot(U_m[i*m+j, :], V_m[j*p+k, :] * W_m[k*n+i, :])
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            try:
                # Use 'lm' (Levenberg-Marquardt) or 'trf'
                # 'lm' is faster for well-posed problems but doesn't support bounds
                # We don't need bounds if we just want to find a real solution
                # then snap to lattice
                sol = least_squares(residuals, x0, method='lm', max_nfev=200)
            except Exception:
                continue
                
            if sol.success or np.allclose(sol.fun, 0, atol=1e-5):
                # Check if close to integer/half-integer
                x_sol = sol.x
                U_sol = x_sol[:n*m*r].reshape(n*m, r)
                V_sol = x_sol[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_sol = x_sol[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify exactness
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_snap[i*m+j, :], V_snap[j*p+k, :] * W_snap[k*n+i, :])
                            if abs(val - 1.0) > 1e-9:
                                exact = False
                                break
                        if not exact:
                            break
                    if not exact:
                        break
                
                if exact:
                    best_result = {'U': U_snap.tolist(), 'V': V_snap.tolist(), 'W': W_snap.tolist()}
                    best_rank = r
                    found = True
                    break # Break restart loop, try lower rank
            
        if found:
            continue
        else:
            # If not found for this rank, we can stop if we've tried a few?
            # Or continue?
            # Usually if rank r is impossible, r-1 is also impossible.
            # But numerical search might fail for r even if it exists.
            # Let's try one more rank down if time permits, then stop.
            # For now, just continue the loop.
            pass

    return best_result
