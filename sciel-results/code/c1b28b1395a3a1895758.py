import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Pre-compute tensor T
    # T[i, j, j', k, i', k'] = delta(i, i') * delta(j, j') * delta(k, k')
    # We need sum_r U[i, j, r] * V[j', k, r] * W[i', k', r] == T[i, j, j', k, i', k']
    
    # To be fast, we can iterate over the 576 equations
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            computed = 0.0
                            for r in range(rank):
                                computed += U_np[i*m+j, r] * V_np[jp*p+k, r] * W_np[kp*n+ip, r]
                            if abs(computed - expected) > 1e-9:
                                return False
    return True

def _calculate_error(U, V, W, n, m, p):
    """Calculate the sum of squared errors."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            computed = 0.0
                            for r in range(rank):
                                computed += U_np[i*m+j, r] * V_np[jp*p+k, r] * W_np[kp*n+ip, r]
                            error += (computed - expected) ** 2
    return error

def _local_search(U, V, W, n, m, p, max_steps=5000, time_limit=10.0):
    """Perform local search on half-integer lattice to minimize error."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    current_error = _calculate_error(U_np, V_np, W_np, n, m, p)
    if current_error < 1e-9:
        return U_np, V_np, W_np, current_error
    
    start_time = time.time()
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    best_error = current_error
    
    # Pre-compute indices for affected equations
    # For each variable, store the list of (i, j, jp, k, ip, kp) it affects
    # U[i, j, r] affects equations (i, j, jp, k, ip, kp) for all jp, k, ip, kp
    # V[jp, k, r] affects equations (i, j, jp, k, ip, kp) for all i, j, ip, kp
    # W[ip, kp, r] affects equations (i, j, jp, k, ip, kp) for all i, j, jp, k
    
    # To speed up, we can just recalculate error from scratch if it's fast enough?
    # 576 equations * 24 multiplications = ~14k ops per error calc.
    # 5000 steps * 14k ops = 70M ops. In Python, this might be slow.
    # Let's use numpy for vectorized error calculation.
    
    # Vectorized error calculation
    def calc_err_vec(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # We want sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
        # This is a tensor contraction.
        # Let's reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # Compute T[i, j, jp, k, ip, kp] = sum_r U_r[i, j, r] * V_r[jp, k, r] * W_r[ip, kp, r]
        # This is equivalent to:
        # Temp1[i, j, jp, k] = sum_r U_r[i, j, r] * V_r[jp, k, r]
        # Temp2[i, j, jp, k, ip, kp] = sum_r ...
        
        # Using einsum
        # T = np.einsum('ijr,jkr,ikr->ijjik', U_r, V_r, W_r) # Indices might be wrong
        # U: i, j, r
        # V: jp, k, r  (Note: V index is jp, k. In equation, V is indexed by jp, k)
        # W: ip, kp, r (Note: W index is ip, kp. In equation, W is indexed by ip, kp)
        
        # The sum is over r.
        # Result indices: i, j, jp, k, ip, kp
        T = np.einsum('ijr,jkr,ikr->ijjik', U_r, V_r, W_r)
        
        # Target tensor
        # T_target[i, j, jp, k, ip, kp] = 1 if i==ip, j==jp, k==kp else 0
        # We can construct this or compute error directly.
        
        # Error = sum (T - T_target)^2
        # T_target is sparse.
        # Non-zero terms are when i==ip, j==jp, k==kp.
        # There are n*m*p = 24 such terms.
        # For these terms, T_target = 1.
        # For all other terms, T_target = 0.
        
        # Sum of squares of T_target = 24 * 1^2 = 24.
        # Sum of T * T_target = sum_{i,j,k} T[i, j, i, k, i, k]
        # Sum of T^2 = np.sum(T**2)
        
        # Error = np.sum(T**2) - 2 * sum(T * T_target) + sum(T_target**2)
        # sum(T * T_target) = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} T[i, j, i, k, i, k]
        
        # Extract the diagonal terms
        diag_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diag_sum += T[i, j, i, k, i, k]
                    
        error = np.sum(T**2) - 2 * diag_sum + 24
        return error

    # Re-initialize with vectorized calc
    current_error = calc_err_vec(U_np, V_np, W_np)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random variable
        # 3 types: U, V, W
        var_type = random.randint(0, 2)
        if var_type == 0:
            idx = (random.randint(0, n*m-1), random.randint(0, rank-1))
            old_val = U_np[idx]
        elif var_type == 1:
            idx = (random.randint(0, m*p-1), random.randint(0, rank-1))
            old_val = V_np[idx]
        else:
            idx = (random.randint(0, p*n-1), random.randint(0, rank-1))
            old_val = W_np[idx]
        
        # Try changing by +/- 0.5 or +/- 1
        delta = random.choice([-1.0, -0.5, 0.5, 1.0])
        new_val = old_val + delta
        
        # Apply change
        if var_type == 0:
            U_np[idx] = new_val
        elif var_type == 1:
            V_np[idx] = new_val
        else:
            W_np[idx] = new_val
            
        new_error = calc_err_vec(U_np, V_np, W_np)
        
        if new_error < current_error:
            current_error = new_error
            if new_error < 1e-9:
                return U_np, V_np, W_np, new_error
        else:
            # Revert
            if var_type == 0:
                U_np[idx] = old_val
            elif var_type == 1:
                V_np[idx] = old_val
            else:
                W_np[idx] = old_val
                
    return U_np, V_np, W_np, current_error

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    rank_schoolbook = n * m * p
    
    # Initialize schoolbook
    U = [[0] * rank_schoolbook for _ in range(dim_U)]
    V = [[0] * rank_schoolbook for _ in range(dim_V)]
    W = [[0] * rank_schoolbook for _ in range(dim_W)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find lower rank
    # We search from low rank up
    # Lower bound is max(n*m, m*p, p*n) = 12
    for target_rank in range(12, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Try a few restarts
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Random initialization
            U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_U, target_rank))
            V_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_V, target_rank))
            W_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_W, target_rank))
            
            # Local search
            U_opt, V_opt, W_opt, err = _local_search(U_rand, V_rand, W_rand, n, m, p, max_steps=2000, time_limit=5.0)
            
            if err < 1e-9:
                # Verify
                if _verify_exact(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                    break # Found valid for this rank, try lower? No, we are going up. So break to next rank?
                    # Actually, if we found rank R, we should try R-1?
                    # The loop goes 12, 13, ...
                    # So if we find 15, we should continue to 16? No, we want MINIMAL.
                    # So if we find a solution for R, we should break and return?
                    # No, maybe 14 works? We started at 12.
                    # If 12 failed, 13 failed, 14 succeeded, then 14 is the best so far.
                    # We should break the restart loop and return?
                    # Or continue to check if 13 works with more restarts?
                    # Given time limit, if we find one, it's good.
                    # But we are iterating R upwards.
                    # So if R=12 fails, we try R=13.
                    # If R=13 succeeds, we record it.
                    # We should break the restart loop for this R, but continue to next R?
                    # No, if R=13 works, we don't need R=14.
                    # So we should return immediately?
                    # Or maybe we can find R=12 with more restarts?
                    # We already did 10 restarts for R=12.
                    # So if R=13 works, it's the best we found.
                    # We can return it.
                    
                    # However, it's possible that R=12 works but we missed it.
                    # But we can't spend infinite time.
                    # Let's return the best found.
                    pass
        # If we found a solution for this target_rank, we can break the outer loop?
        # Since we are going from low to high, the first one we find is the minimum.
        if best_rank == target_rank:
            break
            
    return best_result
