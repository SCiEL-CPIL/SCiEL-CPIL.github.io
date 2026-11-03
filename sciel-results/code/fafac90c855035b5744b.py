import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Trivial solution
    rank_trivial = n * m * p
    U_trivial = np.zeros((dim_u, rank_trivial))
    V_trivial = np.zeros((dim_v, rank_trivial))
    W_trivial = np.zeros((dim_w, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_trivial[u_idx, col] = 1.0
                V_trivial[v_idx, col] = 1.0
                W_trivial[w_idx, col] = 1.0
                col += 1

    # Target tensor entries
    # T[a, b, c] = 1 if (a//m == c%n) and (a%m == b//p) and (b%p == c//n) else 0
    # We only need the non-zero entries for efficiency in the linear system construction
    # but for ALS we might need the whole system or just the non-zeros + zeros.
    # The zeros are important to prevent "leakage".
    
    # List of non-zero indices (a, b, c)
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
                
    # Total number of entries
    total_entries = dim_u * dim_v * dim_w
    
    # Try to find a lower rank solution using ALS
    # We'll try ranks from 24 down to 16
    best_solution = None
    best_rank = rank_trivial
    best_error = np.inf
    
    # Time limit check
    start_time = time.time()
    time_limit = 110  # Leave some margin for 120s total
    
    for rank in range(24, 15, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random initializations
        found = False
        for init in range(5):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + init)
            U = np.random.randn(dim_u, rank) * 0.1
            V = np.random.randn(dim_v, rank) * 0.1
            W = np.random.randn(dim_w, rank) * 0.1
            
            # ALS iterations
            for it in range(50):
                # Update U
                # Minimize || T - sum_r U_r V_r W_r ||^2
                # Linear in U
                # We construct the system for non-zero and zero entries
                # To save memory/time, we can sample zeros or use all if small enough
                # 1728 equations is small.
                
                # Build A_U and y_U
                # y is T
                # A rows correspond to (a,b,c)
                # A[row, :] = [V[b,r] * W[c,r] for all a, r]
                
                # This is expensive to build every time.
                # Let's precompute the structure?
                # Actually, for each a, the row in A is determined by b,c.
                # Let's just build it.
                
                A_U = np.zeros((total_entries, dim_u * rank))
                y_U = np.zeros(total_entries)
                
                idx = 0
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            # Determine T[a,b,c]
                            i = a // m
                            j = a % m
                            j_v = b // p
                            k = b % p
                            k_w = c // n
                            i_w = c % n
                            
                            if i == i_w and j == j_v and k == k_w:
                                y_U[idx] = 1.0
                            else:
                                y_U[idx] = 0.0
                            
                            # Row in A_U for this (a,b,c)
                            # U is flattened: U[a, r] is at index a*rank + r
                            for r in range(rank):
                                A_U[idx, a * rank + r] = V[b, r] * W[c, r]
                            idx += 1
                            
                # Solve least squares
                try:
                    U_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_U, y_U, rcond=None)
                    U = U_flat.reshape(dim_u, rank)
                except:
                    break
                    
                # Update V
                A_V = np.zeros((total_entries, dim_v * rank))
                y_V = np.zeros(total_entries)
                idx = 0
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            i = a // m
                            j = a % m
                            j_v = b // p
                            k = b % p
                            k_w = c // n
                            i_w = c % n
                            
                            if i == i_w and j == j_v and k == k_w:
                                y_V[idx] = 1.0
                            else:
                                y_V[idx] = 0.0
                            
                            for r in range(rank):
                                A_V[idx, b * rank + r] = U[a, r] * W[c, r]
                            idx += 1
                            
                try:
                    V_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_V, y_V, rcond=None)
                    V = V_flat.reshape(dim_v, rank)
                except:
                    break
                    
                # Update W
                A_W = np.zeros((total_entries, dim_w * rank))
                y_W = np.zeros(total_entries)
                idx = 0
                for a in range(dim_u):
                    for b in range(dim_v):
                        for c in range(dim_w):
                            i = a // m
                            j = a % m
                            j_v = b // p
                            k = b % p
                            k_w = c // n
                            i_w = c % n
                            
                            if i == i_w and j == j_v and k == k_w:
                                y_W[idx] = 1.0
                            else:
                                y_W[idx] = 0.0
                            
                            for r in range(rank):
                                A_W[idx, c * rank + r] = U[a, r] * V[b, r]
                            idx += 1
                            
                try:
                    W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_W, y_W, rcond=None)
                    W = W_flat.reshape(dim_w, rank)
                except:
                    break
                    
                # Check error
                # Compute residual
                err = 0.0
                for a, b, c in nz_indices:
                    val = np.dot(U[a, :], V[b, :]) * W[c, :] # No, this is not right
                    # val = sum_r U[a,r] V[b,r] W[c,r]
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err += (val - 1.0)**2
                    
                # Sample some zeros to check leakage
                # Just check a few random zeros
                for _ in range(100):
                    a = np.random.randint(dim_u)
                    b = np.random.randint(dim_v)
                    c = np.random.randint(dim_w)
                    i = a // m; j = a % m
                    j_v = b // p; k = b % p
                    k_w = c // n; i_w = c % n
                    if not (i == i_w and j == j_v and k == k_w):
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        err += val**2
                        
                if err < 1e-6:
                    best_solution = (U, V, W)
                    best_rank = rank
                    found = True
                    break
                    
        if found:
            break

    if best_solution is not None:
        U, V, W = best_solution
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return trivial solution
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
# EVOLVE-BLOCK-END
