"""Matrix multiplication 3x5x6 rank-minimal decomposition."""
DIMENSIONS = [3, 5, 6]

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 3, 5, 6
    
    # Build the tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Start with schoolbook (rank 90)
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
    
    # Try to find a lower rank decomposition using ALS
    def try_rank_decomposition(T, R, max_iters=300, seed=42):
        np.random.seed(seed)
        n1, n2, n3 = T.shape
        
        U = np.random.randn(n1, R) * 0.1
        V = np.random.randn(n2, R) * 0.1
        W = np.random.randn(n3, R) * 0.1
        
        best_err = float('inf')
        U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            VW = np.zeros((n2*n3, R))
            for r in range(R):
                VW[:, r] = (V[:, r] * W[:, r]).flatten()
            for a in range(n1):
                rhs = T[a, :, :].flatten()
                try:
                    U[a, :] = np.linalg.lstsq(VW, rhs, rcond=None)[0]
                except:
                    pass
            
            # Update V
            UW = np.zeros((n1*n3, R))
            for r in range(R):
                UW[:, r] = (U[:, r] * W[:, r]).flatten()
            for b in range(n2):
                rhs = T[:, b, :].flatten()
                try:
                    V[b, :] = np.linalg.lstsq(UW, rhs, rcond=None)[0]
                except:
                    pass
            
            # Update W
            UV = np.zeros((n1*n2, R))
            for r in range(R):
                UV[:, r] = (U[:, r] * V[:, r]).flatten()
            for c in range(n3):
                rhs = T[:, :, c].flatten()
                try:
                    W[c, :] = np.linalg.lstsq(UV, rhs, rcond=None)[0]
                except:
                    pass
            
            # Compute error
            err = 0.0
            for r in range(R):
                contrib = np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                err += np.sum((T - contrib)**2)
            
            if err < best_err:
                best_err = err
                U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-8:
                break
        
        return U_best, V_best, W_best, best_err
    
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to reduce rank
    for R in range(89, 82, -1):
        if time.time() > deadline:
            break
        
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U_try, V_try, W_try, err = try_rank_decomposition(T, R, max_iters=150, seed=seed)
            
            if err < 1e-6:
                # Verify
                T_approx = np.zeros_like(T, dtype=np.float64)
                for r in range(R):
                    T_approx += np.einsum('a,b,c->abc', U_try[:,r], V_try[:,r], W_try[:,r])
                
                if np.max(np.abs(T_approx - T)) < 1e-4:
                    # Check for integer/half-integer entries
                    all_vals = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
                    rounded2 = np.round(all_vals * 2) / 2
                    if np.max(np.abs(all_vals - rounded2)) < 1e-3:
                        U_try = np.round(U_try * 2) / 2
                        V_try = np.round(V_try * 2) / 2
                        W_try = np.round(W_try * 2) / 2
                        best_U, best_V, best_W, best_rank = U_try, V_try, W_try, R
                        break
        
        if best_rank < rank_sb:
            break
    
    def to_clean_list(arr):
        result = []
        for row in arr:
            new_row = []
            for val in row:
                v = float(val)
                if abs(v - round(v)) < 1e-6:
                    new_row.append(int(round(v)))
                elif abs(v * 2 - round(v * 2)) < 1e-6:
                    half = round(v * 2) / 2
                    new_row.append(half if half != int(half) else int(half))
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    return {"U": to_clean_list(best_U), "V": to_clean_list(best_V), "W": to_clean_list(best_W)}

