"""Matrix multiplication 3x4 * 4x8 with numerical tensor decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # The matrix multiplication tensor T is of size (n, m, p) with T[i, j, k] = 1 if i,j,k are valid indices
    # Wait, the standard definition for matrix multiplication tensor in CP decomposition context:
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k].
    # So T is a 3-way tensor of size (n, p, m) or (n, m, p)?
    # The interface requires:
    # U: (n*m, rank) -> corresponds to A indices (i, j)
    # V: (m*p, rank) -> corresponds to B indices (j, k)
    # W: (p*n, rank) -> corresponds to C indices (k, i) note: W[k*n+i]
    #
    # The tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    #
    # Let's reshape this into a standard 3-way tensor form for decomposition.
    # Let X be a tensor of shape (n*m, m*p, p*n).
    # X[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent?
    # No, the sum is over r. The target tensor T has entries T[a, b, c] where:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # The condition is that the indices (i,j) from a, (j,k) from b, and (k,i) from c must match.
    # Specifically, if a = i*m+j, b = j'*p+k, c = k'*n+i', then T[a,b,c] = 1 if i=i', j=j', k=k'.
    # Otherwise T[a,b,c] = 0.
    
    # Dimensions:
    dim_A = n * m  # 12
    dim_B = m * p  # 32
    dim_C = p * n  # 24
    
    # Construct the target tensor T of shape (dim_A, dim_B, dim_C)
    T = np.zeros((dim_A, dim_B, dim_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                T[idx_a, idx_b, idx_c] = 1.0

    # We want to find rank-r vectors U (dim_A x r), V (dim_B x r), W (dim_C x r)
    # such that T approx (or equals) sum_r u_r v_r^T w_r^T (outer product)
    # This is the CP decomposition.
    
    # Strategy:
    # 1. Try small ranks. The theoretical lower bound is max(n, m, p) = 8? No, rank of matmul is at least max(n,m,p).
    # Actually, rank of matmul tensor is known to be at least max(n,m,p).
    # For 3x4x8, schoolbook rank is 96. Strassen-like is 88.
    # Can we do better?
    # Let's try to find a decomposition for ranks starting from 10 up to 88.
    # Since exact decomposition with small rank is hard to find numerically due to sparsity,
    # we might need to use a solver that handles sparse tensors or use a heuristic.
    
    # However, the problem asks for EXACT decomposition with integer/half-integer entries.
    # Numerical optimization will yield floating point. We can round to nearest half-integer.
    # If the numerical solution is close enough, rounding might work.
    
    # Let's use a simple iterative method: Alternating Least Squares (ALS) for CP decomposition.
    # Or, since we suspect the rank is low, we can try to initialize with random vectors and refine.
    
    # Given the time limit, let's try a few ranks.
    # If we can't find an exact one, we fallback to the known 88 rank construction.
    
    def solve_cp_decomposition(tensor, rank, max_iters=100, tol=1e-10):
        """
        Simple ALS for CP decomposition.
        tensor: 3D numpy array
        rank: int
        Returns: U, V, W matrices
        """
        d1, d2, d3 = tensor.shape
        
        # Initialize with random vectors
        U = np.random.randn(d1, rank)
        V = np.random.randn(d2, rank)
        W = np.random.randn(d3, rank)
        
        # Normalize columns to avoid scale ambiguity
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10:
                U[:, r] /= norm_u
            if norm_v > 1e-10:
                V[:, r] /= norm_v
            if norm_w > 1e-10:
                W[:, r] /= norm_w
        
        for it in range(max_iters):
            # Update U
            # For each r, U[:, r] is the solution to min ||tensor - sum_s u_s v_s^T w_s^T||_F
            # This is equivalent to: for each r, U[:, r] = vec(T *_{2} V[:, r] *_{3} W[:, r])
            # Where *_{2} and *_{3} are mode-2 and mode-3 products.
            
            U_old = U.copy()
            for r in range(rank):
                # Compute T contracted with V[:, r] and W[:, r]
                # T_{i, j, k}
                # V[:, r] is length d2
                # W[:, r] is length d3
                # Result is length d1
                # Res = sum_{j,k} T[i, j, k] V[j, r] W[k, r]
                # This is T *_{2} V[:, r] *_{3} W[:, r]
                
                # Efficient computation:
                # T_mat = T.reshape(d1, d2*d3)
                # VW_mat = np.outer(V[:, r], W[:, r]).reshape(d2*d3)
                # U[:, r] = T_mat.T @ VW_mat
                
                # But T is sparse. Let's use the fact that T has only n*m*p non-zeros.
                # We can precompute the non-zero indices.
                # For now, let's just do the dense operation if dimensions are small enough.
                # d1=12, d2=32, d3=24. T size 12*32*24 = 9216. Very small.
                
                T_mat = T.reshape(d1, d2 * d3)
                VW_vec = np.outer(V[:, r], W[:, r]).reshape(-1)
                U[:, r] = T_mat.T @ VW_vec
                
                # Normalize
                norm = np.linalg.norm(U[:, r])
                if norm > 1e-10:
                    U[:, r] /= norm
            
            # Update V
            V_old = V.copy()
            for r in range(rank):
                # V[:, r] = vec(T *_{1} U[:, r] *_{3} W[:, r])
                T_mat = T.reshape(d2, d1 * d3) # Need to move dim 2 to front
                # T is (d1, d2, d3). We want to contract dim 1 (index i) and dim 3 (index k).
                # T *_{1} U[:, r] -> tensor of shape (d2, d3)
                # Let's do it via transpose
                T_trans = np.transpose(T, (1, 0, 2)) # (d2, d1, d3)
                T_mat = T_trans.reshape(d2, d1 * d3)
                UW_vec = np.outer(U[:, r], W[:, r]).reshape(-1)
                V[:, r] = T_mat.T @ UW_vec
                
                norm = np.linalg.norm(V[:, r])
                if norm > 1e-10:
                    V[:, r] /= norm
            
            # Update W
            W_old = W.copy()
            for r in range(rank):
                # W[:, r] = vec(T *_{1} U[:, r] *_{2} V[:, r])
                T_trans = np.transpose(T, (2, 0, 1)) # (d3, d1, d2)
                T_mat = T_trans.reshape(d3, d1 * d2)
                UV_vec = np.outer(U[:, r], V[:, r]).reshape(-1)
                W[:, r] = T_mat.T @ UV_vec
                
                norm = np.linalg.norm(W[:, r])
                if norm > 1e-10:
                    W[:, r] /= norm
            
            # Check convergence
            if np.linalg.norm(U - U_old) < tol and np.linalg.norm(V - V_old) < tol and np.linalg.norm(W - W_old) < tol:
                break
        
        # Reconstruct and check error
        # Reconstruct T_hat
        # T_hat[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # We can compute this efficiently
        T_hat = np.zeros((d1, d2, d3))
        for r in range(rank):
            T_hat += np.outer(U[:, r], V[:, r])[:, :, None] * W[None, None, :]
        
        error = np.linalg.norm(T - T_hat) / np.linalg.norm(T)
        return U, V, W, error

    # Try to find a low rank decomposition
    best_rank = None
    best_U, best_V, best_W = None, None, None
    best_error = float('inf')
    
    # The rank of the matmul tensor is at least max(n,m,p) = 8.
    # It is known that for 2x2x2, rank is 7.
    # For 3x4x8, let's try ranks from 10 to 50.
    # If we find one with very low error, we round it.
    
    for rank in range(10, 51):
        if time.time() > deadline - 10:
            break
        
        # Use multiple restarts for small ranks
        restarts = 3 if rank < 20 else 1
        
        for _ in range(restarts):
            U, V, W, error = solve_cp_decomposition(T, rank, max_iters=50)
            
            if error < best_error:
                best_error = error
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                
                # If error is extremely small, we might have an exact solution
                if error < 1e-8:
                    # Round to half-integers
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify exactness with integer/half-integer values
                    T_hat = np.zeros((dim_A, dim_B, dim_C))
                    for r in range(rank):
                        T_hat += np.outer(U_int[:, r], V_int[:, r])[:, :, None] * W_int[None, None, :]
                    
                    # Check if T_hat equals T
                    if np.allclose(T_hat, T, atol=1e-9):
                        # Success!
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
            
            if time.time() > deadline - 5:
                break
    
    # If no low rank exact solution found, fallback to the known 88 rank construction
    # We reuse the logic from the previous program for the fallback to ensure correctness.
    
    rank_total = 88
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0

    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,0):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    ]

    def add_strassen(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        nonlocal col
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            for (ar, ac), coeff in a_coeffs.items():
                U[(a_row_base + ar) * m + (a_col_base + ac)][col] += coeff
            for (br, bc), coeff in b_coeffs.items():
                V[(b_row_base + br) * p + (b_col_base + bc)][col] += coeff
            for (cr, cc), coeff in c_coeffs.items():
                W[(c_col_base + cc) * n + (c_row_base + cr)][col] += coeff
            col += 1
        return 7

    def add_schoolbook_1x2x2(a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        nonlocal col
        terms = [
            ({(0,0):1}, {(0,0):1}, {(0,0):1}),
            ({(0,1):1}, {(1,0):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1}, {(0,1):1}),
            ({(0,1):1}, {(1,1):1}, {(0,1):1}),
        ]
        for a_coeffs, b_coeffs, c_coeffs in terms:
            for (ar, ac), coeff in a_coeffs.items():
                U[(a_row_base + ar) * m + (a_col_base + ac)][col] += coeff
            for (br, bc), coeff in b_coeffs.items():
                V[(b_row_base + br) * p + (b_col_base + bc)][col] += coeff
            for (cr, cc), coeff in c_coeffs.items():
                W[(c_col_base + cc) * n + (c_row_base + cr)][col] += coeff
            col += 1
        return 4

    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_strassen(0, 0, 0, 2, 0, 2)
    add_strassen(0, 2, 2, 2, 0, 2)
    add_strassen(0, 0, 0, 4, 0, 4)
    add_strassen(0, 2, 2, 4, 0, 4)
    add_strassen(0, 0, 0, 6, 0, 6)
    add_strassen(0, 2, 2, 6, 0, 6)

    add_schoolbook_1x2x2(2, 0, 0, 0, 2, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2, 2, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2, 2, 2)
    add_schoolbook_1x2x2(2, 0, 0, 4, 2, 4)
    add_schoolbook_1x2x2(2, 2, 2, 4, 2, 4)
    add_schoolbook_1x2x2(2, 0, 0, 6, 2, 6)
    add_schoolbook_1x2x2(2, 2, 2, 6, 2, 6)

    assert col == rank_total, f"Expected {rank_total} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

