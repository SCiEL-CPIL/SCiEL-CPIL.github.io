"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W satisfy the bilinear constraints."""
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = np.sum(W[k*n+i] * U[i*m+j1] * V[j2*p+k])
                        target = 1.0 if j1 == j2 else 0.0
                        err = abs(s - target)
                        if err > max_err:
                            max_err = err
        return max_err
    
    # Schoolbook fallback
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
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Strategy: Build a structured decomposition using block decomposition of the 9 columns
    # Partition p=9 into blocks: e.g., [3,3,3] or [4,5] or [2,2,2,3]
    # For each block, use a lower-dimensional MM decomposition
    
    def build_block_decomposition(block_sizes, seed=42):
        """Build a structured decomposition by partitioning the p dimension."""
        np.random.seed(seed)
        total_rank = 0
        U_parts = []
        V_parts = []
        W_parts = []
        
        for block_size in block_sizes:
            # For a 4x4 * 4xblock_size multiplication
            # Use a simple structured approach
            # Rank for this block: n * m * block_size = 4*4*block_size
            # But we can do better with structured terms
            
            # Simple structured: for each (i,j,k) in block, one term
            # This gives rank = 4*4*block_size for this block
            block_rank = n * m * block_size
            
            U_block = np.zeros((n*m, block_rank))
            V_block = np.zeros((m*p, block_rank))
            W_block = np.zeros((p*n, block_rank))
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(block_size):
                        U_block[i*m+j, col] = 1
                        V_block[j*p+k, col] = 1
                        W_block[k*n+i, col] = 1
                        col += 1
            
            U_parts.append(U_block)
            V_parts.append(V_block)
            W_parts.append(W_block)
            total_rank += block_rank
        
        U = np.hstack(U_parts)
        V = np.hstack(V_parts)
        W = np.hstack(W_parts)
        
        return U, V, W, total_rank
    
    # Try various block partitions
    # For p=9: [9], [4,5], [3,3,3], [2,2,2,3], [1,1,1,1,1,1,1,1,1]
    block_partitions = [
        [9],
        [4, 5],
        [5, 4],
        [3, 3, 3],
        [2, 2, 2, 3],
        [2, 3, 2, 2],
        [1, 2, 3, 3],
    ]
    
    for partition in block_partitions:
        if sum(partition) != p:
            continue
        if time.time() > deadline - 20:
            break
        
        U, V, W, r = build_block_decomposition(partition)
        err = check_decomposition(U, V, W, tol=1e-8)
        if err < 1e-6 and r < best_rank:
            best_rank = r
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to optimize from the best structured solution
    # Use a different approach: perturb and refine with coordinate descent
    
    if best_rank < rank_sb:
        U_best = np.array(best_result["U"])
        V_best = np.array(best_result["V"])
        W_best = np.array(best_result["W"])
    else:
        U_best = U_sb.copy()
        V_best = V_sb.copy()
        W_best = W_sb.copy()
    
    # Try rank reduction via structured perturbation
    # Key idea: for rectangular MM, we can sometimes share terms across blocks
    
    # Alternative: try a completely different structured approach
    # Use the fact that C[i,k] = sum_j A[i,j] B[j,k]
    # Group by i: for fixed i, we need to compute 9 values from 4 inputs
    # This is a 4->9 linear map for each i
    
    # Try: for each i (4 choices), decompose the 4->9 map
    # The 4->9 map has matrix B (4x9), and we multiply by row i of A
    # Structured: use 4 basis vectors for the 4-dim input space
    
    # Actually, let me try a more aggressive structured approach:
    # Use the identity that for MM, we can write:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j (A[i,j]) (B[j,k])
    # 
    # A structured rank-1 decomposition uses terms of the form:
    # U[i*m+j, c] = a_{i,j}^{(c)}, V[j*p+k, c] = b_{j,k}^{(c)}, W[k*n+i, c] = w_{i,k}^{(c)}
    # with sum_c a_{i,j}^{(c)} * b_{j',k}^{(c)} * w_{i,k}^{(c)} = delta_{j,j'}
    
    # For a lower bound: the tensor rank is at least max(n*p, m*p, n*m) = max(36,36,16) = 36
    # Known results: rank(4x4x4 MM) = 49 (Laderman for 3x3 is 23, for 4x4 it's known to be between 49 and 64)
    # For 4x4x9, the rank should be higher
    
    # Let me try a direct optimization with better initialization
    # Use the structure: initialize with the schoolbook pattern but with shared structure
    
    def structured_init(rank, seed=42):
        """Create a structured initial guess."""
        np.random.seed(seed)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with a structured pattern
        # For each term c, assign it to a (i,j,k) group
        # But with some sharing
        
        # Simple: use the first n*m*p terms as schoolbook, rest as small random
        for c in range(min(rank, n*m*p)):
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
        
        # Add small random perturbation to break symmetry
        if rank > n*m*p:
            for c in range(n*m*p, rank):
                U[:, c] = np.random.randn(n*m) * 0.01
                V[:, c] = np.random.randn(m*p) * 0.01
                W[:, c] = np.random.randn(p*n) * 0.01
        
        # Even for schoolbook-sized rank, add small perturbation
        U += np.random.randn(n*m, rank) * 0.001
        V += np.random.randn(m*p, rank) * 0.001
        W += np.random.randn(p*n, rank) * 0.001
        
        return U, V, W
    
    def optimize_als(U, V, W, max_sweeps=200, tol=1e-7):
        """Alternating least squares optimization."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 15:
                break
            
            # Optimize U
            for i in range(n):
                for j1 in range(m):
                    A_rows = []
                    b_vals = []
                    for k in range(p):
                        for j2 in range(m):
                            a_vec = V[j2*p+k] * W[k*n+i]
                            A_rows.append(a_vec)
                            b_vals.append(1.0 if j1 == j2 else 0.0)
                    A_mat = np.array(A_rows)
                    b_vec = np.array(b_vals)
                    try:
                        u_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        U[i*m+j1] = u_new
                    except:
                        pass
            
            if time.time() > deadline - 15:
                break
            
            # Optimize V
            for j2 in range(m):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for i in range(n):
                        for j1 in range(m):
                            a_vec = U[i*m+j1] * W[k*n+i]
                            A_rows.append(a_vec)
                            b_vals.append(1.0 if j1 == j2 else 0.0)
                    A_mat = np.array(A_rows)
                    b_vec = np.array(b_vals)
                    try:
                        v_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        V[j2*p+k] = v_new
                    except:
                        pass
            
            if time.time() > deadline - 15:
                break
            
            # Optimize W
            for i in range(n):
                for k in range(p):
                    A_rows = []
                    b_vals = []
                    for j1 in range(m):
                        for j2 in range(m):
                            a_vec = U[i*m+j1] * V[j2*p+k]
                            A_rows.append(a_vec)
                            b_vals.append(1.0 if j1 == j2 else 0.0)
                    A_mat = np.array(A_rows)
                    b_vec = np.array(b_vals)
                    try:
                        w_new, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        W[k*n+i] = w_new
                    except:
                        pass
            
            # Check convergence
            err = check_decomposition(U, V, W, tol=1e-5)
            if err < 1e-7:
                break
        
        return U, V, W, check_decomposition(U, V, W)
    
    # Try various ranks with structured initialization
    # Lower bound is 36, but realistically for 4x4x9 it might be around 60-100
    ranks_to_try = [40, 45, 50, 55, 60, 64, 70, 75, 80, 85, 90, 95, 100]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        if rank >= best_rank:
            continue
        
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 25:
                break
            
            U_init, V_init, W_init = structured_init(rank, seed)
            U_opt, V_opt, W_opt, err = optimize_als(U_init, V_init, W_opt, max_sweeps=100)
            
            if err < 1e-6 and rank < best_rank:
                best_rank = rank
                best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                break
        
        if best_rank <= rank:
            break
    
    # Final validation
    if best_rank < rank_sb:
        U_final = np.array(best_result["U"])
        V_final = np.array(best_result["V"])
        W_final = np.array(best_result["W"])
        if check_decomposition(U_final, V_final, W_final, tol=1e-6):
            return best_result
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
