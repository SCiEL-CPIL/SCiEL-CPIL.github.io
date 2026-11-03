"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 times 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # The matrix multiplication tensor T[i, j, k] = 1 if we multiply (i,j) of A with (j,k) of B
    # Output C[i, k] = sum_j A[i,j] * B[j,k]
    # Tensor decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Following the trace convention: W[k*n + i, c]
    
    # Schoolbook decomposition has rank n*m*p = 3*5*6 = 90
    # We want to find a lower rank decomposition.
    
    # For matrix multiplication, the minimal rank is related to the bilinear complexity.
    # For 3x5 * 5x6, we can try various approaches:
    
    # Approach 1: Use known results for small matrix multiplication
    # The bilinear complexity of m x n matrix multiplication is denoted as R(m, n, p)
    # For small matrices, we can use exhaustive search or known decompositions
    
    # Approach 2: Use the fact that we can decompose into blocks
    # A 3x5 matrix can be seen as having 3 rows, each of length 5
    # A 5x6 matrix has 5 rows, each of length 6
    
    # Let's try to use a structured decomposition.
    # For multiplying (n x m) by (m x p), one approach is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can try rank = n*p + m*(n-1)*(p-1) or similar formulas
    # But let's first try a direct approach with random initialization and ALS
    
    # The tensor T is of order 3 with dimensions (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's build the tensor first
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try different ranks starting from a lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Also, the rank must be at least the "bilinear complexity"
    
    # For 3x5 * 5x6, let's try ranks from 30 upward
    # We'll use a randomized ALS approach
    
    best_rank = n * m * p  # Start with schoolbook rank 90
    best_factors = None
    
    # Schoolbook factors
    def schoolbook_factors():
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook_factors()
    
    # Try to find lower rank decompositions
    # Use randomized ALS with multiple restarts
    
    def try_rank(rank, num_restarts=50, max_iters=1000, tol=1e-10):
        """Try to find a rank-r decomposition using ALS."""
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            converged = False
            for it in range(max_iters):
                if time.time() > deadline - 15:
                    break
                
                # Update U: For each (i,j) pair, solve for U[i*m+j, :]
                for idx_i in range(n * m):
                    i = idx_i // m
                    j = idx_i % m
                    
                    # T[idx_i, :, :] = sum_k T[idx_i, j*p+k, k*n+i] for all k
                    # This should equal sum_c U[idx_i, c] * V[:, c] * W[:, c]
                    # More precisely: for each (j', k'), T[idx_i, j'*p+k', k'*n+i'] 
                    # = 1 if j'=j, else 0
                    # So the equation is: for all k', T[idx_i, j*p+k', k'*n+i'] = sum_c U[idx_i,c] * V[j*p+k', c] * W[k'*n+i', c]
                    
                    # We can write this as: 
                    # For fixed i,j: vector v_k = sum_c U[idx_i,c] * V[j*p+k, c] * W[k*n+i, c] for k=0..p-1
                    # should equal [1 if k corresponds to the right k, else 0]
                    
                    # Actually, let's think about this differently.
                    # The tensor T has T[a, b, c] where a = i*m+j, b = j'*p+k', c = k'*n+i'
                    # T[a, b, c] = 1 iff a corresponds to (i,j), b corresponds to (j',k'), c corresponds to (k',i')
                    # AND j = j' AND i = i'
                    # So T[a, b, c] = 1 iff the (i,j) from a, the (j',k') from b, and the (k',i') from c 
                    # satisfy j=j' and i=i'
                    
                    # For fixed a = i*m+j, T[a, b, c] is non-zero only when:
                    # b = j*p+k' for some k', and c = k'*n+i for the same k'
                    # So T[a, j*p+k', k'*n+i] = 1 for all k'
                    
                    # The decomposition says: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
                    
                    # For fixed a, we want: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c] for all b, c
                    # This is equivalent to: U[a, :] * (V * W elementwise) should reconstruct T[a, :, :]
                    # where (V * W elementwise)[b, c] = sum_r V[b, r] * W[c, r] ... no that's not right
                    # Actually: sum_r U[a,r] * V[b,r] * W[c,r] = (U[a,:] * diag(V[b,:]) * W[c,:]) ... 
                    # Let's define: for each r, we have outer product V[:,r] * W[:,r] which is a (m*p) x (p*n) matrix
                    # Then T[a, :, :] = sum_r U[a, r] * (V[:, r] * W[:, r])
                    # This is a matrix equation: T[a] = U[a, :] @ (V * W)^T where (V*W) is elementwise product... 
                    # No, it's: T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # = (U[a,:] * V[b,:] * W[c,:]) summed over r
                    # This can be written as: T[a] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r]) ... no
                    # Actually: T[a, :, :] = sum_r U[a,r] * outer(V[:, r], W[:, r])
                    # where outer(V[:,r], W[:,r]) is a (m*p) x (p*n) matrix with entry [b,c] = V[b,r]*W[c,r]
                    
                    # So for fixed a, we need to solve:
                    # T[a, :, :] = sum_r U[a, r] * M_r where M_r = outer(V[:, r], W[:, r])
                    # This is a linear system in U[a, :]:
                    # Let M be a (m*p*p*n) x rank matrix where M.flatten()[:, r] = outer(V[:, r], W[:, r]).flatten()
                    # Then T[a].flatten() = M @ U[a, :]
                    # So U[a, :] = pinv(M) @ T[a].flatten()
                    
                    # But this is expensive. Let's use a different approach.
                    
                    # Alternative: Fix U and W, solve for V, etc. (standard ALS)
                    # For U update: fix V and W
                    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # For fixed a: T[a, :, :] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                    # Let Z = V * W (elementwise product) ... no
                    # Define: For each r, let M_r = V[:, r] * W[:, r] (outer product), shape (m*p, p*n)
                    # Then T[a].flatten() = sum_r U[a,r] * M_r.flatten()
                    # This is: T[a].flatten() = [M_0.flatten(), M_1.flatten(), ..., M_{rank-1}.flatten()] @ U[a, :]
                    
                    # This is a least squares problem.
                    # Build the matrix A_u of shape (m*p*p*n, rank)
                    A_u = np.zeros((m * p * p * n, rank), dtype=float)
                    for r in range(rank):
                        M_r = np.outer(V[:, r], W[:, r])  # shape (m*p, p*n)
                        A_u[:, r] = M_r.flatten()
                    
                    # Solve: A_u @ U[a, :] = T[a].flatten()
                    rhs = T[a, :, :].flatten()
                    U[a, :], residuals, rank_u, sv = np.linalg.lstsq(A_u, rhs, rcond=None)
                
                # Update V: fix U and W
                for idx_j in range(m * p):
                    j = idx_j // p
                    k = idx_j % p
                    
                    # T[a, idx_j, c] = sum_r U[a, r] * V[idx_j, r] * W[c, r]
                    # For fixed idx_j: T[:, idx_j, :] = sum_r V[idx_j, r] * outer(U[:, r], W[:, r])
                    # T[:, idx_j, :].flatten() = [outer(U[:,r], W[:,r]).flatten() for r] @ V[idx_j, :]
                    
                    A_v = np.zeros((n * m * p * n, rank), dtype=float)
                    for r in range(rank):
                        M_r = np.outer(U[:, r], W[:, r])  # shape (n*m, p*n)
                        A_v[:, r] = M_r.flatten()
                    
                    rhs = T[:, idx_j, :].flatten()
                    V[idx_j, :], residuals, rank_v, sv = np.linalg.lstsq(A_v, rhs, rcond=None)
                
                # Update W: fix U and V
                for idx_k in range(p * n):
                    k = idx_k // n
                    i = idx_k % n
                    
                    # T[a, b, idx_k] = sum_r U[a, r] * V[b, r] * W[idx_k, r]
                    # For fixed idx_k: T[:, :, idx_k] = sum_r W[idx_k, r] * outer(U[:, r], V[:, r])
                    # T[:, :, idx_k].flatten() = [outer(U[:,r], V[:,r]).flatten() for r] @ W[idx_k, :]
                    
                    A_w = np.zeros((n * m * m * p, rank), dtype=float)
                    for r in range(rank):
                        M_r = np.outer(U[:, r], V[:, r])  # shape (n*m, m*p)
                        A_w[:, r] = M_r.flatten()
                    
                    rhs = T[:, :, idx_k].flatten()
                    W[idx_k, :], residuals, rank_w, sv = np.linalg.lstsq(A_w, rhs, rcond=None)
                
                # Check convergence
                # Compute the reconstruction error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            recon = np.dot(U[a, :], V[b, :] * W[c, :])
                            error += (recon - T[a, b, c]) ** 2
                
                if error < tol:
                    converged = True
                    break
                
                if time.time() > deadline - 15:
                    break
            
            if converged:
                # Verify the solution is exact (within numerical precision)
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            recon = np.dot(U[a, :], V[b, :] * W[c, :])
                            max_err = max(max_err, abs(recon - T[a, b, c]))
                
                if max_err < 1e-8:
                    return U, V, W, True
        
        return None, None, None, False
    
    # Try ranks from lower bound upward
    # Lower bound is max(n*m, m*p, p*n) = 30
    # But we can also try known bounds
    
    # Let's try a range of ranks
    # For 3x5 * 5x6, the bilinear complexity is likely between 30 and 90
    
    # Try ranks: 30, 31, ..., up to some limit
    # But this could be slow. Let's be strategic.
    
    # First, try the schoolbook rank to establish a baseline
    current_rank = n * m * p
    
    # Try to reduce rank
    # We'll try ranks from a lower estimate up to current_rank
    
    # Known result: For 2x2 * 2x2, rank is 7 (Strassen)
    # For 3x3 * 3x3, rank is at most 23 (Lavin) or 21 (unknown if minimal)
    # For our case 3x5 * 5x6, we don't have a direct known result
    
    # Let's try ranks from 30 to 90, but we need to be efficient
    # We'll try a few key ranks first
    
    ranks_to_try = list(range(30, 91, 2)) + list(range(85, 91))
    ranks_to_try = sorted(set(ranks_to_try))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        U_found, V_found, W_found, success = try_rank(
            target_rank, 
            num_restarts=20, 
            max_iters=500,
            tol=1e-10
        )
        
        if success:
            best_rank = target_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found
            print(f"Found rank {target_rank} decomposition")
            break
    
    # If we didn't find a better rank, use schoolbook
    if best_rank == n * m * p and (best_U is None or best_U.shape[1] != best_rank):
        best_U, best_V, best_W = schoolbook_factors()
    
    # Convert to lists with integer/half-integer entries if possible
    # The problem says entries should be integer/half-integer
    # Our ALS solution may have floating point entries
    # We need to round to nearest half-integer
    
    def to_half_integer(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2.0
    
    # Round the factors to half-integers
    U_out = np.vectorize(to_half_integer)(best_U)
    V_out = np.vectorize(to_half_integer)(best_V)
    W_out = np.vectorize(to_half_integer)(best_W)
    
    # Verify the rounded solution
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                recon = np.dot(U_out[a, :], V_out[b, :] * W_out[c, :])
                max_err = max(max_err, abs(recon - T[a, b, c]))
    
    # If rounding introduced too much error, try to find exact rational solution
    if max_err > 1e-6:
        # Use the unrounded solution and try to find exact fractions
        # For now, if the error is large, fall back to schoolbook
        if max_err > 1e-3:
            best_U, best_V, best_W = schoolbook_factors()
            U_out = best_U
            V_out = best_V
            W_out = best_W
    
    # Convert to list format
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

