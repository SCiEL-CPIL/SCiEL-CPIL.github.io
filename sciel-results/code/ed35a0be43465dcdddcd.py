import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    deadline = time.time() + 110  # 110 seconds safety margin
    
    # Precompute indices for fast access
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor T: (n, m, p) with T[i,j,k]=1
    
    best_result = None
    best_rank = target_rank
    
    # Schoolbook solution
    U_school = [[0] * target_rank for _ in range(n * m)]
    V_school = [[0] * target_rank for _ in range(m * p)]
    W_school = [[0] * target_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 2 # Scaled by 2? No, let's keep unscaled for schoolbook or scale consistently.
                # Let's stick to the scaled integer formulation for the search.
                # For schoolbook, we can just return the unscaled version if we want, 
                # but the interface accepts integer/half-integer.
                # Let's generate the schoolbook in the original scale.
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Search for lower ranks
    # We search r from 50 down to 40
    for r in range(50, 39, -1):
        if time.time() > deadline:
            break
        
        # Try a few restarts for this rank
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W with random small integers (scaled by 2)
            # Values in {-2, -1, 0, 1, 2}
            U = np.random.randint(-2, 3, (n * m, r))
            V = np.random.randint(-2, 3, (m * p, r))
            W = np.random.randint(-2, 3, (p * n, r))
            
            # Compute initial tensor product P (scaled)
            # P[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # We use numpy for speed
            # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
            U_reshaped = U.reshape(n, m, r)
            V_reshaped = V.reshape(m, p, r)
            W_reshaped = W.reshape(p, n, r)
            
            # Compute P using einsum
            # P[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
            # Note: W is indexed by k*n+i, so W_reshaped[k, i, c]
            P = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            # Target T is 8 (scaled by 8? No, if U,V,W are scaled by 2, product is scaled by 8)
            # T[i,j,k] = 1. Scaled target = 8.
            T_target = np.ones((n, m, p)) * 8
            
            # Calculate error
            error = np.sum((T_target - P) ** 2)
            
            if error == 0:
                # Found a solution!
                # Convert back to original scale (divide by 2)
                U_out = (U / 2).tolist()
                V_out = (V / 2).tolist()
                W_out = (W / 2).tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
            
            # Local search
            # We will perform a fixed number of steps
            steps = 500
            for _ in range(steps):
                if time.time() > deadline:
                    break
                
                # Pick a random matrix and a random entry
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, r - 1)
                    # Update U[row, col]
                    # Delta P[i,j,k] = delta * V[j*p+k, col] * W[k*n+i, col]
                    # i = row // m, j = row % m
                    i = row // m
                    j = row % m
                    # For all k in 0..p-1:
                    # P[i,j,k] changes by delta * V[j*p+k, col] * W[k*n+i, col]
                    # We need to update error efficiently.
                    # Let's just recalculate the affected slice of P and the error change.
                    
                    # Current P slice for i,j
                    P_slice = P[i, j, :]
                    # Delta P slice
                    # V[j*p + k, col] for k in 0..p-1
                    V_vec = V[j*p : j*p+p, col]
                    # W[k*n + i, col] for k in 0..p-1
                    W_vec = W[0 : p*n : n, col] # Step is n
                    # Wait, W index is k*n + i.
                    # k=0 -> i, k=1 -> n+i, ...
                    # So W_vec = W[i : i + p*n : n]
                    W_vec = W[i : i + p*n : n]
                    
                    Delta_P = V_vec * W_vec
                    
                    # Try delta = +1 and -1
                    best_delta = 0
                    best_new_error = error
                    current_error_slice = np.sum((T_target[i, j, :] - P_slice) ** 2)
                    
                    for delta in [-1, 1]:
                        new_P_slice = P_slice + delta * Delta_P
                        new_error_slice = np.sum((T_target[i, j, :] - new_P_slice) ** 2)
                        # Total error change is new_error_slice - current_error_slice
                        # But other parts of P are unchanged, so total new error = error - current_error_slice + new_error_slice
                        new_total_error = error - current_error_slice + new_error_slice
                        if new_total_error < best_new_error:
                            best_new_error = new_total_error
                            best_delta = delta
                    
                    if best_delta != 0:
                        U[row, col] += best_delta
                        P[i, j, :] += best_delta * Delta_P
                        error = best_new_error
                        
                        if error == 0:
                            U_out = (U / 2).tolist()
                            V_out = (V / 2).tolist()
                            W_out = (W / 2).tolist()
                            return {"U": U_out, "V": V_out, "W": W_out}
                            
                elif mat_idx == 1:
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, r - 1)
                    # Update V[row, col]
                    # j = row // p, k = row % p
                    j = row // p
                    k = row % p
                    # P[i,j,k] changes by delta * U[i*m+j, col] * W[k*n+i, col]
                    # For all i in 0..n-1:
                    # P[i,j,k] changes by delta * U[i*m+j, col] * W[k*n+i, col]
                    U_vec = U[0 : n*m : m, col] # i=0 -> j, i=1 -> m+j ...
                    # U index is i*m + j.
                    # i=0 -> j, i=1 -> m+j, ...
                    U_vec = U[j : j + n*m : m]
                    W_val = W[k * n + i, col] # Wait, i is variable.
                    # W index is k*n + i.
                    # For fixed k, i varies.
                    # W_vec = W[k*n : k*n + n, col]
                    W_vec = W[k * n : k * n + n, col]
                    
                    Delta_P = U_vec * W_vec # Shape (n,)
                    
                    # P slice for j,k is P[:, j, k]
                    P_slice = P[:, j, k]
                    T_slice = T_target[:, j, k]
                    
                    best_delta = 0
                    best_new_error = error
                    current_error_slice = np.sum((T_slice - P_slice) ** 2)
                    
                    for delta in [-1, 1]:
                        new_P_slice = P_slice + delta * Delta_P
                        new_error_slice = np.sum((T_slice - new_P_slice) ** 2)
                        new_total_error = error - current_error_slice + new_error_slice
                        if new_total_error < best_new_error:
                            best_new_error = new_total_error
                            best_delta = delta
                            
                    if best_delta != 0:
                        V[row, col] += best_delta
                        P[:, j, k] += best_delta * Delta_P
                        error = best_new_error
                        
                        if error == 0:
                            U_out = (U / 2).tolist()
                            V_out = (V / 2).tolist()
                            W_out = (W / 2).tolist()
                            return {"U": U_out, "V": V_out, "W": W_out}
                            
                else:
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, r - 1)
                    # Update W[row, col]
                    # k = row // n, i = row % n
                    k = row // n
                    i = row % n
                    # P[i,j,k] changes by delta * U[i*m+j, col] * V[j*p+k, col]
                    # For all j in 0..m-1:
                    # P[i,j,k] changes by delta * U[i*m+j, col] * V[j*p+k, col]
                    U_vec = U[i*m : i*m + m, col]
                    V_vec = V[k : k + m*p : p, col] # j=0 -> k, j=1 -> p+k ...
                    # V index is j*p + k.
                    # j=0 -> k, j=1 -> p+k, ...
                    V_vec = V[k : k + m*p : p]
                    
                    Delta_P = U_vec * V_vec # Shape (m,)
                    
                    # P slice for i,k is P[i, :, k]
                    P_slice = P[i, :, k]
                    T_slice = T_target[i, :, k]
                    
                    best_delta = 0
                    best_new_error = error
                    current_error_slice = np.sum((T_slice - P_slice) ** 2)
                    
                    for delta in [-1, 1]:
                        new_P_slice = P_slice + delta * Delta_P
                        new_error_slice = np.sum((T_slice - new_P_slice) ** 2)
                        new_total_error = error - current_error_slice + new_error_slice
                        if new_total_error < best_new_error:
                            best_new_error = new_total_error
                            best_delta = delta
                            
                    if best_delta != 0:
                        W[row, col] += best_delta
                        P[i, :, k] += best_delta * Delta_P
                        error = best_new_error
                        
                        if error == 0:
                            U_out = (U / 2).tolist()
                            V_out = (V / 2).tolist()
                            W_out = (W / 2).tolist()
                            return {"U": U_out, "V": V_out, "W": W_out}

    return best_result
# EVOLVE-BLOCK-END
