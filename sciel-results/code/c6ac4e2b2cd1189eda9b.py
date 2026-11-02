import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i,j,k
    # We want sum_c U[i, c] * V[j, c] * W[k, c] = 1
    
    # Time limit guard
    start_time = time.time()
    max_time = 110.0 # Leave 10s margin
    
    # Fallback to schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find better rank
    # We search for ranks from 25 to 44
    # For each rank, we do a few random restarts
    
    # Helper to check validity
    def check_validity(U, V, W):
        # U: nm x R, V: mp x R, W: pn x R
        # Convert to numpy
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Compute tensor
        # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
        # This is a 3-way contraction.
        # We can do: M1[i, j, c] = U[i, c] * V[j, c] ? No, dimensions.
        # U is (nm, R), V is (mp, R).
        # Index mapping:
        # U index: i*m + j
        # V index: j*p + k  -- WAIT.
        # The task description says:
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        # Output W[k*n+i, r] -- This is the factor W.
        
        # Let's re-read the index convention carefully.
        # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
        # This implies the factors are aligned with the standard flattening.
        # U[row, c] where row = i*m + j
        # V[row, c] where row = j*p + k
        # W[row, c] where row = k*n + i
        
        # The tensor equation is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i, j, k]
        
        # We need T[i, j, k] = 1 for all i, j, k.
        
        # Let's verify the indices.
        # i in 0..n-1, j in 0..m-1, k in 0..p-1.
        
        # We can construct the target tensor T as a 3D array of ones.
        T = np.ones((n, m, p))
        
        # Compute the decomposition tensor
        # We can iterate over c
        Decomp = np.zeros((n, m, p))
        for c in range(U_np.shape[1]):
            u_col = U_np[:, c]
            v_col = V_np[:, c]
            w_col = W_np[:, c]
            
            # Unflatten
            # u_col corresponds to i, j. shape (n*m,)
            # v_col corresponds to j, k. shape (m*p,)
            # w_col corresponds to k, i. shape (p*n,)
            
            # We need to sum over c of (u[i,j] * v[j,k] * w[k,i])
            
            # This is tricky to vectorize directly without loops over i,j,k if we want to be safe.
            # But n,m,p are small (3,3,5).
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Decomp[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                        
        if np.allclose(Decomp, T):
            # Check if entries are integer or half-integer
            # Multiply by 2 and check if integer
            U_check = np.array(U) * 2
            V_check = np.array(V) * 2
            W_check = np.array(W) * 2
            
            if np.allclose(U_check, np.round(U_check)) and \
               np.allclose(V_check, np.round(V_check)) and \
               np.allclose(W_check, np.round(W_check)):
                return True
        return False

    # Search loop
    # We try ranks 25 to 44
    for R in range(25, 45):
        if time.time() - start_time > max_time:
            break
            
        # Try a few restarts for this rank
        for restart in range(50):
            if time.time() - start_time > max_time:
                break
                
            # Random initialization with half-integers
            # Values in {-1, -0.5, 0, 0.5, 1}
            def rand_half_int():
                return random.choice([-1, -0.5, 0, 0.5, 1])
            
            U = [[rand_half_int() for _ in range(R)] for _ in range(n*m)]
            V = [[rand_half_int() for _ in range(R)] for _ in range(m*p)]
            W = [[rand_half_int() for _ in range(R)] for _ in range(p*n)]
            
            # Local search
            # Calculate initial error
            # We can just check validity directly? No, we need to minimize error.
            # But calculating error is expensive if we do it in pure Python loops.
            # Let's use numpy for error calculation.
            
            def calc_error(U, V, W):
                U_np = np.array(U)
                V_np = np.array(V)
                W_np = np.array(W)
                T = np.ones((n, m, p))
                Decomp = np.zeros((n, m, p))
                for c in range(R):
                    u = U_np[:, c]
                    v = V_np[:, c]
                    w = W_np[:, c]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                Decomp[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
                return np.sum((Decomp - T)**2)
            
            current_err = calc_error(U, V, W)
            
            # If it's already valid, return
            if check_validity(U, V, W):
                return {"U": U, "V": V, "W": W}
                
            # Simple hill climbing
            # We will do a fixed number of steps
            steps = 100
            for step in range(steps):
                if time.time() - start_time > max_time:
                    break
                    
                # Pick a random matrix and entry
                mat_idx = random.randint(0, 2) # 0:U, 1:V, 2:W
                if mat_idx == 0:
                    rows, cols = n*m, R
                elif mat_idx == 1:
                    rows, cols = m*p, R
                else:
                    rows, cols = p*n, R
                    
                r = random.randint(0, rows-1)
                c = random.randint(0, cols-1)
                
                old_val = U[r][c] if mat_idx == 0 else (V[r][c] if mat_idx == 1 else W[r][c])
                
                # Perturb
                # Try adding -0.5, 0.5, 1, -1
                delta = random.choice([-0.5, 0.5, 1.0, -1.0])
                new_val = old_val + delta
                
                # Apply
                if mat_idx == 0:
                    U[r][c] = new_val
                elif mat_idx == 1:
                    V[r][c] = new_val
                else:
                    W[r][c] = new_val
                    
                new_err = calc_error(U, V, W)
                
                if new_err < current_err:
                    current_err = new_err
                    # Check if valid
                    if check_validity(U, V, W):
                        if R < best_rank:
                            best_rank = R
                            best_result = {"U": U, "V": V, "W": W}
                        return best_result
                else:
                    # Revert
                    if mat_idx == 0:
                        U[r][c] = old_val
                    elif mat_idx == 1:
                        V[r][c] = old_val
                    else:
                        W[r][c] = old_val

    return best_result
