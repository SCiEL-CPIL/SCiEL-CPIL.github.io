import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    # Shapes:
    # U: (n*m, R) = (6, R)
    # V: (m*p, R) = (21, R)
    # W: (p*n, R) = (14, R)
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find a better rank over F13
    q = 13
    # Target ranks to try
    target_ranks = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    
    # Define the tensor T over F13
    # T[i, j, k] = 1 if index matches
    # Flatten indices:
    # i in [0, n), j in [0, m), k in [0, p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    T_flat = np.zeros((n*m*p), dtype=int)
    # Actually, the tensor is a 3D array T[i, j, k]
    # But for the equation sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
    # We can flatten the LHS into a linear system for one factor.
    
    # Let's build the "linear system" matrices for each factor update.
    # This is complex to build dynamically. 
    # Instead, we can use a simple iterative refinement:
    # 1. Initialize U, V, W randomly.
    # 2. Compute residual.
    # 3. Update one factor by solving a least squares problem over F13.
    
    def solve_ls_Fq(A, b, q):
        # Solve A x = b over F_q
        # A is (M, N), b is (M,)
        # Returns x if consistent, else None
        # Use Gaussian Elimination
        A = np.array(A) % q
        b = np.array(b) % q
        M, N = A.shape
        # Augmented matrix
        A_aug = np.hstack([A, b.reshape(-1, 1)]) % q
        
        # Gaussian Elimination
        rank = 0
        for col in range(N):
            # Find pivot
            pivot_row = -1
            for row in range(rank, M):
                if A_aug[row, col] != 0:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            
            # Swap rows
            A_aug[[rank, pivot_row]] = A_aug[[pivot_row, rank]]
            
            # Normalize pivot
            inv_pivot = pow(A_aug[rank, col], q-2, q)
            A_aug[rank, :] = (A_aug[rank, :] * inv_pivot) % q
            
            # Eliminate
            for row in range(M):
                if row != rank:
                    factor = A_aug[row, col]
                    if factor != 0:
                        A_aug[row, :] = (A_aug[row, :] - factor * A_aug[rank, :]) % q
            
            rank += 1
        
        # Check consistency
        for row in range(rank, M):
            if A_aug[row, N] != 0:
                return None
        
        # Extract solution (free variables set to 0)
        x = np.zeros(N, dtype=int)
        for row in range(rank):
            # Find pivot column
            for col in range(N):
                if A_aug[row, col] != 0:
                    x[col] = A_aug[row, N]
                    break
        return x

    def check_decomposition(U, V, W, R, n, m, p, q):
        # Verify if U, V, W decompose T
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(R):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    val %= q
                    if val != 1:
                        return False
        return True

    best_result = None
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random init
            U = np.random.randint(0, q, (n*m, R))
            V = np.random.randint(0, q, (m*p, R))
            W = np.random.randint(0, q, (p*n, R))
            
            # Iterative improvement
            improved = True
            iter_count = 0
            while improved and iter_count < 50:
                improved = False
                iter_count += 1
                
                # Update U
                # For each u_index (i*m+j), we want U[u_index, :] to satisfy:
                # sum_r U[u_index, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k?
                # No, the equation is:
                # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
                # This couples i, j, k.
                # If we fix V and W, the equation for U[i*m+j, r] is not linear in U alone because it's a product of 3.
                # It is linear in U if we treat V*W as the "coefficient"?
                # No, V and W are also matrices.
                # The term is U[u, r] * V[v, r] * W[w, r].
                # For a fixed set of indices (i, j, k), we have a sum over r.
                # If we fix V and W, we can form a linear system for U?
                # Let X[r] = V[j*p+k, r] * W[k*n+i, r].
                # Then sum_r U[i*m+j, r] X[r] = 1.
                # But X depends on k.
                # So for a fixed (i, j), we have p equations (one for each k) for the same U row.
                # U[i*m+j, :] is a vector of length R.
                # We have p equations:
                # sum_r U[u, r] * (V[j*p+0, r] * W[0*n+i, r]) = 1
                # sum_r U[u, r] * (V[j*p+1, r] * W[1*n+i, r]) = 1
                # ...
                # sum_r U[u, r] * (V[j*p+p-1, r] * W[(p-1)*n+i, r]) = 1
                
                # This is a linear system A_u * U_u = b_u
                # A_u is (p, R), b_u is (p,) with all 1s.
                
                new_U = np.zeros((n*m, R), dtype=int)
                for i in range(n):
                    for j in range(m):
                        u_idx = i*m + j
                        A_u = np.zeros((p, R), dtype=int)
                        for k in range(p):
                            v_idx = j*p + k
                            w_idx = k*n + i
                            for r in range(R):
                                A_u[k, r] = (V[v_idx, r] * W[w_idx, r]) % q
                        
                        b_u = np.ones(p, dtype=int)
                        sol = solve_ls_Fq(A_u, b_u, q)
                        if sol is not None:
                            new_U[u_idx, :] = sol
                        else:
                            # Inconsistent, keep old or random?
                            new_U[u_idx, :] = U[u_idx, :]
                            improved = False # Maybe stuck
                
                # Check if U changed
                if not np.array_equal(new_U, U):
                    U = new_U
                    improved = True
                else:
                    break # No change in U, likely stuck or converged
                
                # Repeat for V and W if time permits
                if time.time() > deadline:
                    break
                
                # Update V
                new_V = np.zeros((m*p, R), dtype=int)
                for j in range(m):
                    for k in range(p):
                        v_idx = j*p + k
                        A_v = np.zeros((n, R), dtype=int) # One eq per i
                        for i in range(n):
                            u_idx = i*m + j
                            w_idx = k*n + i
                            for r in range(R):
                                A_v[i, r] = (U[u_idx, r] * W[w_idx, r]) % q
                        b_v = np.ones(n, dtype=int)
                        sol = solve_ls_Fq(A_v, b_v, q)
                        if sol is not None:
                            new_V[v_idx, :] = sol
                        else:
                            new_V[v_idx, :] = V[v_idx, :]
                            improved = False
                
                if not np.array_equal(new_V, V):
                    V = new_V
                    improved = True
                else:
                    break
                
                # Update W
                new_W = np.zeros((p*n, R), dtype=int)
                for k in range(p):
                    for i in range(n):
                        w_idx = k*n + i
                        A_w = np.zeros((m, R), dtype=int) # One eq per j
                        for j in range(m):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            for r in range(R):
                                A_w[j, r] = (U[u_idx, r] * V[v_idx, r]) % q
                        b_w = np.ones(m, dtype=int)
                        sol = solve_ls_Fq(A_w, b_w, q)
                        if sol is not None:
                            new_W[w_idx, :] = sol
                        else:
                            new_W[w_idx, :] = W[w_idx, :]
                            improved = False
                
                if not np.array_equal(new_W, W):
                    W = new_W
                    improved = True
                else:
                    break
            
            # Check if valid
            if check_decomposition(U, V, W, R, n, m, p, q):
                # Found a valid decomposition over F13
                # Map to Q (integers are fine)
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break # Found best rank for this R, try smaller R? 
                # Since we iterate R ascending, the first found is minimal.
                # But we might have skipped smaller R if they failed.
                # We should break and return.
        
        if best_result:
            break

    if best_result:
        return best_result
    else:
        # Fallback to schoolbook
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
