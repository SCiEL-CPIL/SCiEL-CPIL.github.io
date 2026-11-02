"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def tensor_error(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(recon - T))
    
    def polish(U, V, W, max_iters=2000, tol=1e-14):
        """High-precision polish pass with adaptive step."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            TW = np.einsum('abc,cr->abr', T, W)
            inner_U = np.einsum('abr,br->ar', TW, V)
            denom_U = np.sum(V**2, axis=0) * np.sum(W**2, axis=0)
            U_new = inner_U / (denom_U + 1e-30)
            
            # Update V
            TU = np.einsum('abc,ar->bcr', T, U_new)
            inner_V = np.einsum('bcr,cr->br', TU, W)
            denom_V = np.sum(U_new**2, axis=0) * np.sum(W**2, axis=0)
            V_new = inner_V / (denom_V + 1e-30)
            
            # Update W
            inner_W = np.einsum('bcr,br->cr', TU, V_new)
            denom_W = np.sum(U_new**2, axis=0) * np.sum(V_new**2, axis=0)
            W_new = inner_W / (denom_W + 1e-30)
            
            U, V, W = U_new, V_new, W_new
            
            if it % 100 == 0 or it == max_iters - 1:
                err = tensor_error(U, V, W)
                if err < tol:
                    break
        
        return U, V, W, tensor_error(U, V, W)
    
    # Try to find rank-20 decomposition using ALS with multiple seeds
    def try_als(rank, max_iters=500, seeds=None):
        if seeds is None:
            seeds = list(range(50))
        
        best_overall = None
        best_overall_err = float('inf')
        
        for seed in seeds:
            if time.time() > deadline - 20:
                break
            
            rng = np.random.RandomState(seed)
            U = rng.randn(n*m, rank)
            V = rng.randn(m*p, rank)
            W = rng.randn(p*n, rank)
            
            # Normalize initial vectors
            for r in range(rank):
                nU = np.linalg.norm(U[:,r])
                nV = np.linalg.norm(V[:,r])
                nW = np.linalg.norm(W[:,r])
                if nU > 1e-15: U[:,r] /= nU
                if nV > 1e-15: V[:,r] /= nV
                if nW > 1e-15: W[:,r] /= nW
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            prev_err = float('inf')
            stall_count = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                TW = np.einsum('abc,cr->abr', T, W)
                inner_U = np.einsum('abr,br->ar', TW, V)
                denom_U = np.sum(V**2, axis=0) * np.sum(W**2, axis=0)
                U = inner_U / (denom_U + 1e-15)
                
                # Update V
                TU = np.einsum('abc,ar->bcr', T, U)
                inner_V = np.einsum('bcr,cr->br', TU, W)
                denom_V = np.sum(U**2, axis=0) * np.sum(W**2, axis=0)
                V = inner_V / (denom_V + 1e-15)
                
                # Update W
                inner_W = np.einsum('bcr,br->cr', TU, V)
                denom_W = np.sum(U**2, axis=0) * np.sum(V**2, axis=0)
                W = inner_W / (denom_W + 1e-15)
                
                if it % 10 == 0 or it == max_iters - 1:
                    err = tensor_error(U, V, W)
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if err < prev_err - 1e-12:
                            stall_count = 0
                        else:
                            stall_count += 1
                        prev_err = err
                    else:
                        stall_count += 1
                    
                    if err < 1e-12:
                        break
                    if stall_count > 10 and it > 50:
                        break
            
            if best_err < best_overall_err:
                best_overall_err = best_err
                best_overall = (best_U, best_V, best_W, best_err)
            
            if best_err < 1e-8:
                # Polish
                U_p, V_p, W_p, err_p = polish(best_U, best_V, best_W)
                if err_p < 1e-10:
                    return U_p, V_p, W_p, err_p
        
        return best_overall
    
    # Try rank 20
    result20 = try_als(20, max_iters=400)
    if result20 is not None:
        U, V, W, err = result20
        if err < 1e-8:
            U_p, V_p, W_p, err_p = polish(U, V, W)
            if err_p < 1e-10:
                return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
            elif err < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 21 Strassen
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: C00 + C11
        [(1, 0, 1), (1, 1, -1)],     # M2: C10 - C11
        [(0, 1, 1), (1, 1, 1)],      # M3: C01 + C11
        [(0, 0, 1), (1, 0, 1)],      # M4: C00 + C10
        [(0, 0, -1), (0, 1, 1)],     # M5: -C00 + C01
        [(1, 1, 1)],                 # M6: C11
        [(0, 0, 1)]                  # M7: C00
    ]
    
    for b in range(3):
        for t in range(7):
            r = b * 7 + t
            for idx, val in U_coeffs[t].items():
                U[idx][r] = val
            for (j, k_local, val) in V_coeffs[t]:
                k_global = 2 * b + k_local
                v_idx = j * p + k_global
                V[v_idx][r] = val
            for (i, k_local, val) in W_coeffs[t]:
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
