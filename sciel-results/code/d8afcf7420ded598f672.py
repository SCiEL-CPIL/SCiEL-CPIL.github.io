import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin before 160s wall
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12
    
    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = (U_fb, V_fb, W_fb)
    
    # Strategy: Fix U and V to be structured, solve for W
    # For each (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    # We can try rank = n * p = 12 (one term per output entry)
    # Or rank = n * m * p / m = n * p if we can share across j
    
    # Try rank = n * p = 12: one column per (i,k) pair
    # U[i*m+j, r(i,k)] = 1 for all j, V[j*p+k, r(i,k)] = 1 for all j
    # Then W[k*n+i, r(i,k)] should be 1, and we need sum_j 1*1 = m = 3, not 1
    # So we need scaling: U[i*m+j, r] = 1/m, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # Then sum_j (1/m)*1 = 1. But we also need zero for other (a,b,c).
    # For fixed (i,k), r(i,k): U[a,r] nonzero only if a = i*m+j for some j
    # V[b,r] nonzero only if b = j'*p+k for some j'
    # W[c,r] nonzero only if c = k'*n+i' with k'=k, i'=i
    # So T_approx[a,b,c] for c=k*n+i: sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_j (1/m) * 1 * 1 = 1. Good!
    # For c != k*n+i: W[c,r] = 0, so T_approx = 0. Good!
    # But wait, we need T_approx[a,b,c] = 0 for (a,b,c) not matching any (i,j,k) triple.
    # For a = i1*m+j1, b = j2*p+k2, c = k3*n+i3:
    # T_approx = sum_r U[a,r]*V[b,r]*W[c,r]
    # Only r where a matches i1, b matches k2, c matches i3 contribute
    # r(i1,k2) has U[a,r] = 1/m if j1 is any j, V[b,r] = 1 if j2=k2... wait no
    # V[b,r(i1,k2)] is nonzero only if b = j*p+k2 for some j, i.e., j2 = k2? No, b = j*p+k, so k2 is the second index
    # Let me re-examine: V index is j*p + k. So V[b, r(i1,k2)] nonzero if b = j*p + k2 for some j.
    # So j2 must equal... b = j2*p + k2, so yes, V[b, r(i1,k2)] = 1 if the k-part of b is k2.
    # U[a, r(i1,k2)] nonzero if a = i1*m + j for some j, i.e., i-part of a is i1.
    # W[c, r(i1,k2)] nonzero if c = k2*n + i1, i.e., c exactly equals k2*n + i1.
    # So for c = k3*n + i3, only r(i1,k2) with i1=i3, k2=k3 contributes.
    # Then T_approx[a,b,c] = U[a, r(i3,k3)] * V[b, r(i3,k3)] * W[c, r(i3,k3)]
    # = (1/m if i-part of a is i3) * (1 if k-part of b is k3) * (1 if c = k3*n+i3)
    # For this to equal T[a,b,c], we need:
    # If a = i3*m+j, b = j*p+k3, c = k3*n+i3: T=1, T_approx = (1/m)*1*1 = 1/m ≠ 1. Bad!
    
    # So this simple approach doesn't work directly. Let me try a different structure.
    
    # Alternative: Use rank = n*m*p = 36 but try to find a lower rank via optimization.
    # Or try a different structured approach.
    
    # Let's try: for each j, create m*p terms that handle the j-th "slice"
    # Actually, let's just try the optimization approach with a good initialization.
    
    # Try ranks from 18 to 35
    for rank in range(18, min(36, rank_schoolbook)):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        for trial in range(20):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            
            # Initialize with small random values
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Use alternating least squares: fix two, solve for third
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For fixed U,V: W[c,r] = sum_{a,b} T[a,b,c]*U[a,r]*V[b,r] / sum_{a,b} U[a,r]^2*V[b,r]^2
            # But this is not exactly right for least squares. Let's do proper ALS.
            
            # Actually, let's use scipy.optimize with a simpler objective
            from scipy.optimize import least_squares
            
            def make_residuals(U, V, W):
                def residuals(params):
                    U_r = params[:nm*rank].reshape(nm, rank)
                    V_r = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_r = params[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Compute T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                    # Efficient: for each r, outer product
                    T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    return (T_approx - T).ravel()
                return residuals
            
            x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            try:
                result = least_squares(make_residuals(U, V, W), x0, method='trf', 
                                       max_nfev=500, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                
                if np.linalg.norm(result.fun) < 1e-6:
                    U_r = result.x[:nm*rank].reshape(nm, rank)
                    V_r = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_r = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Verify
                    T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.max(np.abs(T_approx - T)) < 1e-6:
                        best_rank = rank
                        best_result = (U_r, V_r, W_r)
                        break  # Found valid at this rank
                
            except Exception:
                pass
        
        if best_rank < rank_schoolbook:
            break
    
    # If we found something better, return it
    if best_rank < rank_schoolbook:
        U_r, V_r, W_r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
