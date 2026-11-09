"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses a robust vectorized ALS approach to search for lower ranks, 
    falling back to the schoolbook rank 60 solution.
    """
    start_time = time.time()
    deadline = 140  # Keep 10s safety margin before 150s limit
    n, m, p = 3, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 12, 20, 15
    
    # Schoolbook baseline (Rank 60)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Precompute index arrays for vectorized operations
    # For each (i,j,k) triple, store the indices
    ij_indices = []
    jk_indices = []
    ki_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ij_indices.append(i*m + j)
                jk_indices.append(j*p + k)
                ki_indices.append(k*n + i)
    ij_arr = np.array(ij_indices)
    jk_arr = np.array(jk_indices)
    ki_arr = np.array(ki_indices)
    
    def check_solution_vectorized(U, V, W, rank):
        """Compute max absolute error using vectorized operations."""
        # For each of the n*m*p entries, compute dot(U[ij,:], V[jk,:]*W[ki,:])
        # U[ij_arr, :] has shape (60, rank)
        # V[jk_arr, :] has shape (60, rank)
        # W[ki_arr, :] has shape (60, rank)
        vals = np.sum(U[ij_arr, :rank] * V[jk_arr, :rank] * W[ki_arr, :rank], axis=1)
        return float(np.max(np.abs(vals - 1.0)))
    
    def solve_als(rank, seed, max_iters=2000, time_budget=30.0):
        """
        Solve for U, V, W using ALS with vectorized updates.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(nm, rank) * 0.1
        V = rng.randn(mp, rank) * 0.1
        W = rng.randn(pn, rank) * 0.1
        
        reg = 1e-8
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Precompute index arrays for each update type
        # U update: for each (i,j), need indices into V and W
        u_i = np.repeat(np.arange(n), m*p)  # not quite, let's do it differently
        # Actually, for U[i*m+j], we need for each k: V[j*p+k] and W[k*n+i]
        # Let's precompute for each (i,j) pair
        u_pairs = []
        for i in range(n):
            for j in range(m):
                v_idx = j*p + np.arange(p)
                w_idx = np.arange(p)*n + i
                u_pairs.append((i*m+j, v_idx, w_idx))
        
        # For V[j*p+k], need for each i: U[i*m+j] and W[k*n+i]
        v_pairs = []
        for j in range(m):
            for k in range(p):
                u_idx = np.arange(n)*m + j
                w_idx = k*n + np.arange(n)
                v_pairs.append((j*p+k, u_idx, w_idx))
        
        # For W[k*n+i], need for each j: U[i*m+j] and V[j*p+k]
        w_pairs = []
        for k in range(p):
            for i in range(n):
                u_idx = i*m + np.arange(m)
                v_idx = np.arange(m)*p + k
                w_pairs.append((k*n+i, u_idx, v_idx))
        
        for it in range(max_iters):
            elapsed = time.time() - start_time
            if elapsed > start_time + time_budget:
                break
            
            # Update U
            for (idx, v_idx_arr, w_idx_arr) in u_pairs:
                A_u = V[v_idx_arr, :] * W[w_idx_arr, :]  # (p, rank)
                b_u = np.ones(p)
                try:
                    U[idx, :] = np.linalg.solve(A_u.T @ A_u + reg * np.eye(rank), A_u.T @ b_u)
                except np.linalg.LinAlgError:
                    pass
                        
            # Update V
            for (idx, u_idx_arr, w_idx_arr) in v_pairs:
                A_v = U[u_idx_arr, :] * W[w_idx_arr, :]  # (n, rank)
                b_v = np.ones(n)
                try:
                    V[idx, :] = np.linalg.solve(A_v.T @ A_v + reg * np.eye(rank), A_v.T @ b_v)
                except np.linalg.LinAlgError:
                    pass

            # Update W
            for (idx, u_idx_arr, v_idx_arr) in w_pairs:
                A_w = U[u_idx_arr, :] * V[v_idx_arr, :]  # (m, rank)
                b_w = np.ones(m)
                try:
                    W[idx, :] = np.linalg.solve(A_w.T @ A_w + reg * np.eye(rank), A_w.T @ b_w)
                except np.linalg.LinAlgError:
                    pass
            
            # Check error every 10 iterations
            if it % 10 == 0 or it == max_iters - 1:
                curr_err = check_solution_vectorized(U, V, W, rank)
                if curr_err < best_err:
                    best_err = curr_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-12:
                    break
            
            # Early exit if no improvement in last 100 iterations
            if it > 100 and best_err > 1e-6 and it % 50 == 0:
                # Check if we're stuck
                pass
                
        return best_err, best_U, best_V, best_W
    
    def polish(U, V, W, rank, max_iters=500, time_budget=10.0):
        """High-precision polish pass with tighter tolerance."""
        best_err = check_solution_vectorized(U, V, W, rank)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        reg = 1e-12
        
        # Precompute index arrays
        u_pairs = []
        for i in range(n):
            for j in range(m):
                v_idx = j*p + np.arange(p)
                w_idx = np.arange(p)*n + i
                u_pairs.append((i*m+j, v_idx, w_idx))
        
        v_pairs = []
        for j in range(m):
            for k in range(p):
                u_idx = np.arange(n)*m + j
                w_idx = k*n + np.arange(n)
                v_pairs.append((j*p+k, u_idx, w_idx))
        
        w_pairs = []
        for k in range(p):
            for i in range(n):
                u_idx = i*m + np.arange(m)
                v_idx = np.arange(m)*p + k
                w_pairs.append((k*n+i, u_idx, v_idx))
        
        for it in range(max_iters):
            if time.time() - start_time > start_time + time_budget:
                break
            
            for (idx, v_idx_arr, w_idx_arr) in u_pairs:
                A_u = V[v_idx_arr, :] * W[w_idx_arr, :]
                b_u = np.ones(p)
                try:
                    U[idx, :] = np.linalg.solve(A_u.T @ A_u + reg * np.eye(rank), A_u.T @ b_u)
                except np.linalg.LinAlgError:
                    pass
                        
            for (idx, u_idx_arr, w_idx_arr) in v_pairs:
                A_v = U[u_idx_arr, :] * W[w_idx_arr, :]
                b_v = np.ones(n)
                try:
                    V[idx, :] = np.linalg.solve(A_v.T @ A_v + reg * np.eye(rank), A_v.T @ b_v)
                except np.linalg.LinAlgError:
                    pass

            for (idx, u_idx_arr, v_idx_arr) in w_pairs:
                A_w = U[u_idx_arr, :] * V[v_idx_arr, :]
                b_w = np.ones(m)
                try:
                    W[idx, :] = np.linalg.solve(A_w.T @ A_w + reg * np.eye(rank), A_w.T @ b_w)
                except np.linalg.LinAlgError:
                    pass
            
            curr_err = check_solution_vectorized(U, V, W, rank)
            if curr_err < best_err:
                best_err = curr_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-14:
                break
                
        return best_err, best_U, best_V, best_W
    
    # Try ranks from low to high
    # Known lower bound for 3x4x5 is around 31, best known upper is ~47
    ranks_to_try = [32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    
    for rank in ranks_to_try:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        found = False
        seeds = [42, 123, 999, 2024, 555, 777, 888, 1111, 2222, 3333]
        
        for seed in seeds:
            elapsed = time.time() - start_time
            if elapsed > deadline - 10:
                break
            
            time_left = deadline - elapsed - 5
            if time_left < 2:
                break
            
            # Allocate time: use most of remaining time for this attempt
            # But cap it to allow trying other seeds if this fails
            alloc_time = min(time_left, 25.0)
            
            err, U_c, V_c, W_c = solve_als(rank, seed, max_iters=1000, time_budget=alloc_time)
            
            if err < 1e-10:
                # Polish for higher precision
                time_left2 = deadline - (time.time() - start_time) - 3
                if time_left2 > 2:
                    err2, U_p, V_p, W_p = polish(U_c, V_c, W_c, rank, max_iters=300, time_budget=min(time_left2, 10.0))
                    if err2 < err:
                        err = err2
                        U_c, V_c, W_c = U_p, V_p, W_p
                
                if err < 1e-8:
                    best_rank = rank
                    best_U, best_V, best_W = U_c, V_c, W_c
                    found = True
                    break
        
        if found:
            break
            
    # Final verification
    if best_rank < rank_schoolbook:
        final_err = check_solution_vectorized(best_U, best_V, best_W, best_rank)
        if final_err > 1e-8:
            # Fallback
            best_rank = rank_schoolbook
            best_U, best_V, best_W = U_sb, V_sb, W_sb

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    U_out = to_list(best_U[:, :best_rank])
    V_out = to_list(best_V[:, :best_rank])
    W_out = to_list(best_W[:, :best_rank])
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
