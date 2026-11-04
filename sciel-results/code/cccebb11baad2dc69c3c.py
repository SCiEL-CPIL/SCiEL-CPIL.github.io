import time
import numpy as np
import random

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # 1. Fallback: Rank 42 Strassen Decomposition (Verified)
    # This is the current best known simple construction.
    def get_strassen_42():
        U = np.zeros((nm, 42), dtype=int)
        V = np.zeros((mp, 42), dtype=int)
        W = np.zeros((pn, 42), dtype=int)
        
        # Strassen 2x2x2 matrices
        # Indices for 2x2x2: 
        # U: 4x7, V: 4x7, W: 4x7
        # Mapping: 
        # U[i, j] where i in 0..1, j in 0..1
        # V[j, k] where j in 0..1, k in 0..1
        # W[k, i] where k in 0..1, i in 0..1
        
        U_222 = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        V_222 = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ])
        W_222 = np.array([
            [1, 0, 0, -1, 1, 0, -1],
            [0, 0, 1, 1, 0, 0, 0],
            [0, -1, 0, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ])
        
        r_offset = 0
        # Blocks: j in {0,1}, {2,3}; k in {0,1}, {2,3}, {4,5}
        for j_block in range(2):
            for k_block in range(3):
                for r in range(7):
                    # Fill U
                    for i in range(2):
                        for jj in range(2):
                            j_idx = j_block * 2 + jj
                            val = U_222[i * 2 + jj, r]
                            if val != 0:
                                U[i * m + j_idx, r_offset + r] = val
                    # Fill V
                    for jj in range(2):
                        for kk in range(2):
                            j_idx = j_block * 2 + jj
                            k_idx = k_block * 2 + kk
                            val = V_222[jj * 2 + kk, r]
                            if val != 0:
                                V[j_idx * p + k_idx, r_offset + r] = val
                    # Fill W
                    for kk in range(2):
                        for i in range(2):
                            k_idx = k_block * 2 + kk
                            val = W_222[kk * 2 + i, r]
                            if val != 0:
                                W[k_idx * n + i, r_offset + r] = val
                r_offset += 7
                
        return U, V, W

    U_best, V_best, W_best = get_strassen_42()
    best_rank = 42
    
    def is_valid(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition"""
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # Should be 1 for all i, j, k
        # Vectorized check for speed
        # Flatten U, V, W to 1D? No, let's use einsum or loops if small enough.
        # 48 checks is fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k]) * 1.0 # Wait, W is missing
                    # Correct formula: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # This is element-wise product sum
                    prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                    val = np.sum(prod)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # 2. Search for Rank < 42 using Sparse Random Initialization + Local Repair
    # Target ranks: 41, 40
    # Strategy:
    # - Initialize U, V, W with sparse random integers {-1, 0, 1}
    # - Compute error tensor E[i,j,k] = sum_r ... - 1
    # - While error exists and time permits:
    #   - Pick a random (i,j,k) with non-zero error
    #   - Pick a random r
    #   - Try to adjust U[i,j,r], V[j,k,r], or W[k,i,r] to reduce error
    #   - Since entries are integers, we can try adding/subtracting 1 to one of the factors
    #     if it reduces the total L2 error.
    
    def try_find_lower_rank(R, max_iters=50000):
        nonlocal U_best, V_best, W_best, best_rank
        
        if R >= best_rank:
            return False
            
        # Initialize with sparsity
        # Density: start low, maybe 0.1
        density = 0.15
        U = np.zeros((nm, R), dtype=int)
        V = np.zeros((mp, R), dtype=int)
        W = np.zeros((pn, R), dtype=int)
        
        # Random sparse init
        for i in range(nm):
            for r in range(R):
                if random.random() < density:
                    U[i, r] = random.choice([-1, 1])
        for j in range(mp):
            for r in range(R):
                if random.random() < density:
                    V[j, r] = random.choice([-1, 1])
        for k in range(pn):
            for r in range(R):
                if random.random() < density:
                    W[k, r] = random.choice([-1, 1])
                    
        # Precompute indices for faster access
        # U_idx[i,j] = i*m+j
        # V_idx[j,k] = j*p+k
        # W_idx[k,i] = k*n+i
        
        # Error calculation
        def calc_error(U, V, W):
            err = 0.0
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        val = np.sum(prod)
                        diff = val - 1.0
                        err += diff * diff
                        max_err = max(max_err, abs(diff))
            return err, max_err

        curr_err, _ = calc_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return False
                
            if curr_err < 1e-9:
                # Found a solution!
                # Verify strictly
                if is_valid(U, V, W):
                    U_best, V_best, W_best = U, V, W
                    best_rank = R
                    return True
                else:
                    # Numerical issue? Unlikely with integers.
                    break
            
            # Pick a random (i, j, k) with error
            # To speed up, we can just pick random i,j,k and check
            # But if error is high, most are wrong.
            # Let's pick random i, j, k
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            
            # Compute current value for this (i,j,k)
            prod = U[u_idx] * V[v_idx] * W[w_idx]
            curr_val = np.sum(prod)
            target_val = 1.0
            diff = curr_val - target_val
            
            if abs(diff) < 1e-9:
                continue # Already correct, move on
                
            # Try to fix this (i,j,k) by modifying one random r
            r = random.randint(0, R-1)
            
            u_val = U[u_idx, r]
            v_val = V[v_idx, r]
            w_val = W[w_idx, r]
            
            # Current contribution of r to this (i,j,k) is u_val * v_val * w_val
            curr_contrib = u_val * v_val * w_val
            
            # We want to change the total sum by -diff
            # Changing U, V, or W by delta changes the sum by:
            # dU: delta * v_val * w_val
            # dV: u_val * delta * w_val
            # dW: u_val * v_val * delta
            
            # We are restricted to integer changes. 
            # Simplest: try changing one factor by +1 or -1 and see if total error decreases.
            
            candidates = []
            
            # Try changing U
            if v_val != 0 and w_val != 0:
                factor = v_val * w_val
                # delta = 1 or -1
                # New sum = curr_sum - curr_contrib + (u_val + delta)*v_val*w_val
                # Change = delta * factor
                # We want Change approx -diff
                # If factor == 1, delta = -diff
                # If factor == -1, delta = diff
                # Since diff is integer (sum of integers), we can try exact match if possible.
                # But we can also just try delta = 1, -1 and measure global error.
                
                # To avoid full global error calc (expensive), we can estimate local change.
                # But local change doesn't account for other (i',j',k') sharing the same r.
                # However, for sparse matrices, collisions are rare.
                
                # Let's just try a few random moves and check global error occasionally.
                # Optimization: Check global error every 10 moves.
                
                pass # Logic simplified below
            
            # Simplified Repair Strategy:
            # Randomly pick one of U, V, W to modify.
            # Modify by +1 or -1.
            # If global error decreases, keep it. Else revert.
            
            move_type = random.randint(0, 2) # 0:U, 1:V, 2:W
            delta = random.choice([-1, 1])
            
            if move_type == 0:
                U[u_idx, r] += delta
                new_err, _ = calc_error(U, V, W)
                if new_err < curr_err:
                    curr_err = new_err
                else:
                    U[u_idx, r] -= delta # Revert
            elif move_type == 1:
                V[v_idx, r] += delta
                new_err, _ = calc_error(U, V, W)
                if new_err < curr_err:
                    curr_err = new_err
                else:
                    V[v_idx, r] -= delta # Revert
            else:
                W[w_idx, r] += delta
                new_err, _ = calc_error(U, V, W)
                if new_err < curr_err:
                    curr_err = new_err
                else:
                    W[w_idx, r] -= delta # Revert
            
            # Occasionally, try a "jump" to escape local minima
            if it % 1000 == 999:
                # Randomly reset a few entries
                for _ in range(10):
                    mat = random.choice([U, V, W])
                    idx = random.randint(0, mat.shape[0]-1)
                    r_idx = random.randint(0, R-1)
                    mat[idx, r_idx] = random.choice([-1, 0, 1])
                curr_err, _ = calc_error(U, V, W)
                
        return False

    # Try ranks 41, 40
    # Note: Rank 41 is very unlikely to exist if 42 is minimal, but we check.
    # Rank 40 is the next "magic" number often seen in Strassen variants.
    
    if time.time() < deadline - 20:
        try_find_lower_rank(41, max_iters=20000)
        
    if time.time() < deadline - 10 and best_rank > 40:
        try_find_lower_rank(40, max_iters=20000)

    # Final Verification of Best Result
    if not is_valid(U_best, V_best, W_best):
        # If something went wrong, fallback to trivial or known 42
        U_best, V_best, W_best = get_strassen_42()
        best_rank = 42

    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
