import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 9
    nm = n * m       # 16
    mp = m * p       # 36
    pn = p * n       # 36
    
    # Helper: build full tensor matrix from U, V, W
    # T[r, i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    def tensor_from_UVW(U, V, W):
        r = U.shape[1]
        # Use einsum for speed
        # U is (nm, r), V is (mp, r), W is (pn, r)
        # We want S[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let's reshape U, V, W to 3D
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # S[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        # This is a tensor contraction.
        # We can do this in steps to avoid huge intermediate arrays if possible, 
        # but n,m,p are small (4,4,9). r is up to 144.
        # Intermediate: A[i, j, k, r] = U_3d[i, j, r] * V_3d[j, k, r]
        # A shape: (4, 4, 9, r) -> 144*r. If r=100, 14400. Small.
        A = U_3d[:, :, None, :] * V_3d[None, :, :, :] 
        # A[i, j, k, r]
        # S[i, j, k] = sum_r A[i, j, k, r] * W_3d[k, i, r]
        # W_3d is (p, n, r) -> (9, 4, r)
        # We need to align indices: k from A matches first dim of W, i from A matches second dim of W.
        # W_3d is indexed [k, i, r].
        # So we just multiply A and W_3d and sum over r.
        S = np.einsum('ijk r, k i r -> i j k', A, W_3d)
        return S

    def tensor_residual(U, V, W):
        S = tensor_from_UVW(U, V, W)
        target = np.ones((n, m, p))
        return np.sum((S - target)**2)
    
    # Start with schoolbook decomposition
    rank_sb = n * m * p  # 144
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using numerical optimization
    best_U, best_V, best_W = None, None, None
    best_rank = rank_sb
    
    # We will try to find a solution.
    # If we find a numerical solution, we will try to snap it to half-integers.
    # If the snapped solution is exact, we use it.
    # If not, we keep the best numerical one (but we must return exact).
    # The task requires EXACT decomposition.
    # So if we don't find an exact one, we must return the schoolbook (or another known exact one).
    # Are there other known exact decompositions?
    # For 4x4x4, rank 49 is known (Strassen-like).
    # For 4x4x9, we can embed 4x4x4? No, p=9.
    # We can use the 4x4x4 decomposition for the first 4 columns of the 9?
    # Actually, we can just use the schoolbook for the whole thing.
    # Or, we can construct a block diagonal?
    # If we decompose 4x4x4 with rank 49, and 4x4x5 with rank 80?
    # 49 + 80 = 129 < 144.
    # Is there a rank 49 decomposition for 4x4x4? Yes.
    # Can we find it?
    # The Strassen algorithm is for 2x2.
    # For 4x4, there are known rank 49 decompositions.
    # But finding them numerically is hard.
    
    # Let's stick to the numerical search but improve it.
    # If we find a rank < 144 that snaps to integers/half-integers exactly, we win.
    
    candidate_ranks = [70, 80, 90, 100, 110, 120, 130]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
            
        # Try multiple random initializations
        found_exact = False
        for init_seed in range(5):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(42 + target_rank + init_seed)
            U_init = np.random.randn(nm, target_rank)
            V_init = np.random.randn(mp, target_rank)
            W_init = np.random.randn(pn, target_rank)
            
            U_init /= np.sqrt(nm)
            V_init /= np.sqrt(mp)
            W_init /= np.sqrt(pn)
            
            def flatten_params(x):
                U = x[:nm*target_rank].reshape(nm, target_rank)
                V = x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W = x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                return U, V, W
            
            def objective(x):
                if time.time() > deadline - 10:
                    return 1e10
                U, V, W = flatten_params(x)
                S = tensor_from_UVW(U, V, W)
                return np.sum((S - 1)**2)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B
            # Increase iterations
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if result.fun < 1e-4:
                U_opt, V_opt, W_opt = flatten_params(result.x)
                
                # Try to snap to half-integers
                # Multiply by 2 to check for integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify exactness
                S_snap = tensor_from_UVW(U_snap, V_snap, W_snap)
                if np.allclose(S_snap, 1, atol=1e-9):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = target_rank
                    found_exact = True
                    break
        
        if found_exact:
            break
            
    if best_U is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
