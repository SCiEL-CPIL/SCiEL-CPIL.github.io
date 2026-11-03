import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    # Build T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    deadline = 120.0
    safety = 5.0

    best_rank = n*m*p
    best_U = None
    best_V = None
    best_W = None

    # Schoolbook fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1

    # Try to find lower rank using optimization
    # We try ranks from 41 down to 30
    for rank in range(41, 29, -1):
        if time.time() > deadline - safety - 10:
            break
            
        # Try a few random initializations
        found = False
        for seed in range(5):
            if time.time() > deadline - safety - 5:
                break
                
            rng = np.random.RandomState(seed + rank*100)
            # Initialize with random vectors
            U = rng.randn(nm, rank) * 0.1
            V = rng.randn(mp, rank) * 0.1
            W = rng.randn(pn, rank) * 0.1
            
            def objective(params):
                U_flat, V_flat, W_flat = np.split(params, [nm*rank, nm*rank + mp*rank])
                U = U_flat.reshape(nm, rank)
                V = V_flat.reshape(mp, rank)
                W = W_flat.reshape(pn, rank)
                
                # Compute sum_r u_r v_r w_r
                # Efficient way:
                # P = U @ (V * W) ? No.
                # P[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
                # This is a tensor contraction.
                # We can do:
                # Temp = U[:, :, np.newaxis] * V[np.newaxis, :, :] -> (nm, rank, mp)
                # This is too big.
                
                # Better:
                # Flatten T to 1D.
                # Compute residual directly.
                # P = np.einsum('ur,vr,wr->uvw', U, V, W)
                P = np.einsum('ur,vr,wr->uvw', U, V, W)
                err = np.sum((P - T)**2)
                return err
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                if res.fun < 1e-8:
                    # Found a candidate
                    U_opt, V_opt, W_opt = np.split(res.x, [nm*rank, nm*rank + mp*rank])
                    U_opt = U_opt.reshape(nm, rank)
                    V_opt = V_opt.reshape(mp, rank)
                    W_opt = W_opt.reshape(pn, rank)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    P_snap = np.einsum('ur,vr,wr->uvw', U_snap, V_snap, W_snap)
                    if np.allclose(P_snap, T, atol=1e-9):
                        # Check if entries are half-integers (they are by construction)
                        best_rank = rank
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            # We found a lower rank. Can we go lower?
            # The loop continues to lower ranks.
            pass

    if best_U is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
