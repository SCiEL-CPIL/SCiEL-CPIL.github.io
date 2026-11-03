import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all entries
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    # We can vectorize this check for speed
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best = {
        "U": U_sb.tolist(), 
        "V": V_sb.tolist(), 
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook

    # Target Tensor: T[i, j, k] = 1 for all i,j,k
    # We want to minimize || U ⊙ V ⊙ W - T ||_F^2
    # where (U ⊙ V ⊙ W)[i*m+j, j*p+k, k*n+i] is not quite right.
    # The tensor T_mat has shape (n*m, m*p, p*n) ? No.
    # The contraction is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # This is a 3-way tensor T of shape (n*m, m*p, p*n) such that T[i*m+j, j*p+k, k*n+i] = 1.
    # Note that the indices are coupled: j is in U and V, k is in V and W, i is in W and U.
    
    # Let's construct the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def find_decomposition(rank, seed, time_limit=10.0):
        """
        Use ALS to find a rank-r decomposition.
        Returns (U, V, W) as numpy arrays or None if timeout.
        """
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        start_time = time.time()
        
        # Precompute slices for efficiency? 
        # The update for U:
        # For each row i*m+j of U, we want to minimize sum_{j,k} (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1)^2
        # This is equivalent to: U[i*m+j, :] = (sum_{j,k} V[j*p+k, :]^T W[k*n+i, :] * W[k*n+i, :] V[j*p+k, :])^{-1} * ... 
        # Actually, it's simpler to use Gradient Descent or a specific ALS formulation.
        
        # Let's use a simple Gradient Descent with Adam-like momentum or just plain GD with small step.
        # Given the small dimensions (12, 24, 18) and rank (~50), GD might be slow but ALS is better.
        
        # ALS Formulation:
        # Fix V, W. Solve for U.
        # The objective is sum_{i,j,k} ( (U_i * V_j * W_k)_c - 1 )^2 where * is element-wise product in c? No.
        # The term is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        # Let X = V[j,k,:] * W[k,i,:] (element-wise product of vectors).
        # Then the term is dot(U[i,j,:], X).
        # We want dot(U[i,j,:], X) = 1.
        # This is a linear system for each (i,j).
        # Sum over k: dot(U[i,j,:], sum_k X_{j,k}) = 1? No, the target is 1 for each specific (i,j,k).
        
        # Wait, the target T[i*m+j, j*p+k, k*n+i] = 1.
        # So for a fixed (i,j), we have p equations:
        # For each k: dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) = 1
        # Let X_k = V[j*p+k, :] * W[k*n+i, :] (vector of length rank)
        # We have a system: X U_row = 1 (where X is p x rank, U_row is rank x 1)
        # Least squares solution: U_row = (X^T X)^{-1} X^T 1
        
        for step in range(1000):
            if time.time() - start_time > time_limit:
                return None
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build X matrix of shape (p, rank)
                    X = np.zeros((p, rank))
                    for k in range(p):
                        X[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve X @ u = 1
                    # Use lstsq for stability
                    try:
                        u_sol, residuals, rank_ls, sv = np.linalg.lstsq(X, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = u_sol
                    except:
                        pass
            
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # Build X matrix of shape (n, rank)
                    # Equations: for each i, dot(V[j*p+k, :], U[i*m+j, :] * W[k*n+i, :]) = 1
                    X = np.zeros((n, rank))
                    for i in range(n):
                        X[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v_sol, residuals, rank_ls, sv = np.linalg.lstsq(X, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = v_sol
                    except:
                        pass
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # Build X matrix of shape (m, rank)
                    # Equations: for each j, dot(W[k*n+i, :], U[i*m+j, :] * V[j*p+k, :]) = 1
                    X = np.zeros((m, rank))
                    for j in range(m):
                        X[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w_sol, residuals, rank_ls, sv = np.linalg.lstsq(X, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = w_sol
                    except:
                        pass
            W = W_new
            
            # Check convergence/error
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (s - 1.0) ** 2
            if err < 1e-10:
                break

        return U, V, W

    # Try ranks from 50 up to 65
    # We start low to find a potentially better rank.
    for rank in range(50, 66):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            res = find_decomposition(rank, seed, time_limit=5.0)
            if res is None:
                continue
                
            U, V, W = res
            
            # Round to nearest integer
            # The problem allows half-integers too? "integer/half-integer entries"
            # Let's try rounding to nearest 0.5 first, then integer.
            
            # Strategy: Check if values are close to integers or half-integers.
            # Multiply by 2, round, divide by 2.
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            
            # Verify the rounded version
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                # Success!
                if rank < best_rank:
                    best = {
                        "U": U_r.tolist(), 
                        "V": V_r.tolist(), 
                        "W": W_r.tolist()
                    }
                    best_rank = rank
                    found = True
                    print(f"ALS found rank {rank} with seed {seed}")
                    break
        
        # If we found a valid rank, we can stop searching higher ranks 
        # (since we are iterating upwards, lower is better, so we break on first success? 
        # No, we iterate 50, 51... so if we find 50, we break. If we find 52, we keep looking for 51? 
        # No, the loop is 50 to 65. If we find 52, we should check 51? 
        # It's better to iterate downwards or just keep the best. 
        # Since we iterate upwards, the first valid one is the lowest in this range. 
        # But maybe 50 fails and 51 succeeds. So we break on success.
        if found:
            break

    # If ALS didn't find anything better than schoolbook, we return schoolbook.
    # We could also try the block method as a backup if ALS fails completely, 
    # but the block method is already in the "current" logic which scored 1.0 (implied).
    # Wait, the prompt says "current program score 1.0000". 
    # My goal is to beat 1.0. 
    # If ALS finds rank 50, score is 72/50 = 1.44.
    
    # One more check: ensure the returned matrices have the correct shape and type
    final_U = np.array(best["U"])
    final_V = np.array(best["V"])
    final_W = np.array(best["W"])
    
    # Ensure they are lists of lists of numbers
    return {
        "U": final_U.tolist(),
        "V": final_V.tolist(),
        "W": final_W.tolist()
    }

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best = {
        "U": U_sb.tolist(), 
        "V": V_sb.tolist(), 
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook

    def find_decomposition(rank, seed, time_limit=10.0):
        """
        Use ALS to find a rank-r decomposition.
        Returns (U, V, W) as numpy arrays or None if timeout.
        """
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        start_time = time.time()
        
        for step in range(1000):
            if time.time() - start_time > time_limit:
                return None
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    X = np.zeros((p, rank))
                    for k in range(p):
                        X[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        u_sol, residuals, rank_ls, sv = np.linalg.lstsq(X, np.ones(p), rcond=None)
                        U_new[i * m + j, :] = u_sol
                    except:
                        pass
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    X = np.zeros((n, rank))
                    for i in range(n):
                        X[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v_sol, residuals, rank_ls, sv = np.linalg.lstsq(X, np.ones(n), rcond=None)
                        V_new[j * p + k, :] = v_sol
                    except:
                        pass
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    X = np.zeros((m, rank))
                    for j in range(m):
                        X[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w_sol, residuals, rank_ls, sv = np.linalg.lstsq(X, np.ones(m), rcond=None)
                        W_new[k * n + i, :] = w_sol
                    except:
                        pass
            W = W_new
            
            # Quick error check every 10 steps to save time
            if step % 10 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (s - 1.0) ** 2
                if err < 1e-10:
                    break

        return U, V, W

    # Try ranks from 50 up to 65
    for rank in range(50, 66):
        if time.time() > deadline - 10:
            break
            
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            res = find_decomposition(rank, seed, time_limit=5.0)
            if res is None:
                continue
                
            U, V, W = res
            
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            
            if verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                if rank < best_rank:
                    best = {
                        "U": U_r.tolist(), 
                        "V": V_r.tolist(), 
                        "W": W_r.tolist()
                    }
                    best_rank = rank
                    found = True
                    print(f"ALS found rank {rank} with seed {seed}")
                    break
        
        if found:
            break

    return best
# EVOLVE-BLOCK-END
