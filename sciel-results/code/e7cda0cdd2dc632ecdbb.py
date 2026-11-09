import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # U size: n*m x r
    # V size: m*p x r
    # W size: p*n x r
    
    # Target tensor T
    # T[a, b, c] where a in 0..nm-1, b in 0..mp-1, c in 0..pn-1
    # T[a, b, c] = 1 if a=im+j, b=jp+k, c=pn+i? No.
    # W index is k*n + i.
    # U index is i*m + j.
    # V index is j*p + k.
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    start_time = time.time()
    time_limit = 100.0 # Leave margin
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 15 up to 24
    for r in range(15, best_rank + 1):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a solution of rank r
        # Use random restarts
        found = False
        restarts = 0
        max_restarts = 50 if r == 15 else 10
        
        while not found and restarts < max_restarts:
            if time.time() - start_time > time_limit:
                break
                
            restarts += 1
            # Initialize U, V, W with random integers -1, 0, 1
            U = np.random.randint(-1, 2, (n*m, r))
            V = np.random.randint(-1, 2, (m*p, r))
            W = np.random.randint(-1, 2, (p*n, r))
            
            # Local search
            # Compute current error
            # Tensor product is expensive if done naively.
            # We can compute the error efficiently?
            # E = sum (T - U V^T W)^2 is not quite right because it's a 3-tensor.
            # We can flatten and use einsum or just loops for small sizes.
            
            def calc_error(U, V, W):
                # Reconstruct T'
                # T'[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
                # We can compute this as:
                # M1 = U @ V.T  -> shape (nm, mp, r)? No.
                # U (nm, r), V (mp, r).
                # U @ V.T is (nm, mp, r) if we broadcast?
                # U[:, None, :] * V[None, :, :] -> (nm, mp, r)
                # Then dot with W?
                # (nm, mp, r) * W[None, None, :] -> (nm, mp, pn, r)
                # Sum over r -> (nm, mp, pn)
                
                # This creates a large array (6, 12, 8, r).
                # 6*12*8*15 = 8640. Very small.
                prod = U[:, None, None, :] * V[None, :, None, :] * W[None, None, :, :]
                T_recon = np.sum(prod, axis=3)
                return np.sum((T - T_recon) ** 2)
            
            err = calc_error(U, V, W)
            
            # Simple hill climbing
            improved = True
            while improved and (time.time() - start_time) < time_limit:
                improved = False
                # Try changing a random entry
                # Choose random matrix, row, col
                mat_idx = np.random.randint(0, 3)
                if mat_idx == 0:
                    row = np.random.randint(0, n*m)
                    col = np.random.randint(0, r)
                    old_val = U[row, col]
                    # Try new values
                    for new_val in range(-2, 3):
                        if new_val == old_val: continue
                        U[row, col] = new_val
                        new_err = calc_error(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                        else:
                            U[row, col] = old_val
                elif mat_idx == 1:
                    row = np.random.randint(0, m*p)
                    col = np.random.randint(0, r)
                    old_val = V[row, col]
                    for new_val in range(-2, 3):
                        if new_val == old_val: continue
                        V[row, col] = new_val
                        new_err = calc_error(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                        else:
                            V[row, col] = old_val
                else:
                    row = np.random.randint(0, p*n)
                    col = np.random.randint(0, r)
                    old_val = W[row, col]
                    for new_val in range(-2, 3):
                        if new_val == old_val: continue
                        W[row, col] = new_val
                        new_err = calc_error(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                        else:
                            W[row, col] = old_val
                
                if err == 0:
                    found = True
                    break
                    
        if found:
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            break # Found minimal rank in this sequence
            
    if best_U is None:
        # Fallback to schoolbook
        best_rank = n * m * p
        U = np.zeros((n*m, best_rank))
        V = np.zeros((m*p, best_rank))
        W = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        best_U = U
        best_V = V
        best_W = W
        
    # Convert to lists of lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
