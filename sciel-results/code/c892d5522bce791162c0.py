"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 6x6x6
    # Split into 2x2 blocks of size 3x3
    block_size = 3
    num_blocks = 2
    
    # 7 Strassen terms
    # Each term is a product of two (block_size x block_size) matrices
    # Rank of each term is block_size^3 = 27
    # Total rank = 7 * 27 = 189
    
    rank = 7 * (block_size ** 3)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for A blocks in X_s
    # s: term index 0..6
    # a, b: block row, col of A
    alpha = {}
    # s=0: A00 + A11
    alpha[(0, 0, 0)] = 1; alpha[(0, 1, 1)] = 1
    # s=1: A10 + A11
    alpha[(1, 1, 0)] = 1; alpha[(1, 1, 1)] = 1
    # s=2: A00
    alpha[(2, 0, 0)] = 1
    # s=3: A11
    alpha[(3, 1, 1)] = 1
    # s=4: A00 + A01
    alpha[(4, 0, 0)] = 1; alpha[(4, 0, 1)] = 1
    # s=5: A10 - A00
    alpha[(5, 1, 0)] = 1; alpha[(5, 0, 0)] = -1
    # s=6: A01 - A11
    alpha[(6, 0, 1)] = 1; alpha[(6, 1, 1)] = -1
    
    # Coefficients for B blocks in Y_s
    # s: term index 0..6
    # c, d: block row, col of B
    gamma = {}
    # s=0: B00 + B11
    gamma[(0, 0, 0)] = 1; gamma[(0, 1, 1)] = 1
    # s=1: B00
    gamma[(1, 0, 0)] = 1
    # s=2: B01 - B11
    gamma[(2, 0, 1)] = 1; gamma[(2, 1, 1)] = -1
    # s=3: B10 - B00
    gamma[(3, 1, 0)] = 1; gamma[(3, 0, 0)] = -1
    # s=4: B11
    gamma[(4, 1, 1)] = 1
    # s=5: B00 + B01
    gamma[(5, 0, 0)] = 1; gamma[(5, 0, 1)] = 1
    # s=6: B10 + B11
    gamma[(6, 1, 0)] = 1; gamma[(6, 1, 1)] = 1
    
    # Coefficients for M_s in C blocks
    # s: term index 0..6
    # i1, k1: block row, col of C
    beta = {}
    # s=0 (M1): C00 + C11
    beta[(0, 0, 0)] = 1; beta[(0, 1, 1)] = 1
    # s=1 (M2): C10 - C11
    beta[(1, 1, 0)] = 1; beta[(1, 1, 1)] = -1
    # s=2 (M3): C01 + C11
    beta[(2, 0, 1)] = 1; beta[(2, 1, 1)] = 1
    # s=3 (M4): C00 + C10
    beta[(3, 0, 0)] = 1; beta[(3, 1, 0)] = 1
    # s=4 (M5): -C00 + C01
    beta[(4, 0, 0)] = -1; beta[(4, 0, 1)] = 1
    # s=5 (M6): C11
    beta[(5, 1, 1)] = 1
    # s=6 (M7): C00
    beta[(6, 0, 0)] = 1
    
    # Construct U, V, W
    # r index: s * 27 + i0 * 9 + j0 * 3 + k0
    for s in range(7):
        for i0 in range(block_size):
            for j0 in range(block_size):
                for k0 in range(block_size):
                    r = s * (block_size ** 3) + i0 * (block_size ** 2) + j0 * block_size + k0
                    
                    # U: A part
                    for a in range(num_blocks):
                        for b in range(num_blocks):
                            coeff = alpha.get((s, a, b), 0)
                            if coeff != 0:
                                i = a * block_size + i0
                                j = b * block_size + j0
                                U[i * m + j][r] = coeff
                                
                    # V: B part
                    for c in range(num_blocks):
                        for d in range(num_blocks):
                            coeff = gamma.get((s, c, d), 0)
                            if coeff != 0:
                                j = c * block_size + j0
                                k = d * block_size + k0
                                V[j * p + k][r] = coeff
                                
                    # W: C part
                    for i1 in range(num_blocks):
                        for k1 in range(num_blocks):
                            coeff = beta.get((s, i1, k1), 0)
                            if coeff != 0:
                                i = i1 * block_size + i0
                                k = k1 * block_size + k0
                                W[k * n + i][r] = coeff
                                
    # Attempt rank reduction via SVD/Linear Dependency Check
    # Flatten rank-1 tensors to check for linear dependencies
    # T_r = u_r \otimes v_r \otimes w_r
    # If the vectors t_r are linearly dependent, we might reduce rank.
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check rank of the set of rank-1 tensors
    # Construct matrix where each column is a flattened tensor
    # Shape: (n*m*p, rank)
    flat_tensors = np.zeros((n * m * p, rank))
    
    for r in range(rank):
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        
        # Outer product u \otimes v \otimes w
        # Index in flat tensor: (i, j, k) -> i * m * p + j * p + k ?
        # Task says W[k*n+i, r].
        # U is (n*m, rank). Index i*m+j.
        # V is (m*p, rank). Index j*p+k.
        # W is (p*n, rank). Index k*n+i.
        
        # We need to map the tensor indices (i, j, k) to the flat vector.
        # The tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # To check dependencies, we flatten the tensor T into a vector of length n*m*p.
        # Let's construct the vector for each r.
        vec_r = np.zeros(n * m * p)
        
        # Iterate over non-zeros in u_col, v_col, w_col?
        # Or just do the outer product logic.
        # Since U, V, W are sparse-ish (mostly 0, 1, -1), we can be efficient.
        
        # For each r, the tensor is u_r \otimes v_r \otimes w_r.
        # We can fill vec_r by iterating over non-zero entries.
        
        # Get indices of non-zeros
        u_idx = np.nonzero(u_col)[0]
        v_idx = np.nonzero(v_col)[0]
        w_idx = np.nonzero(w_col)[0]
        
        for ui in u_idx:
            i, j = divmod(ui, m)
            u_val = u_col[ui]
            for vi in v_idx:
                j2, k = divmod(vi, p)
                if j != j2: continue # Must match j
                v_val = v_col[vi]
                for wi in w_idx:
                    k2, i2 = divmod(wi, n)
                    if k != k2 or i != i2: continue # Must match k and i
                    w_val = w_col[wi]
                    
                    # If consistent, add to vec_r
                    flat_idx = i * m * p + j * p + k
                    vec_r[flat_idx] = u_val * v_val * w_val
        
        flat_tensors[:, r] = vec_r
    
    # Compute rank of flat_tensors
    # Use SVD to find effective rank
    try:
        # SVD can be expensive for 216x189, but it's small enough.
        U_svd, s_vals, Vt_svd = np.linalg.svd(flat_tensors, full_matrices=False)
        # Threshold for rank
        tol = 1e-9
        effective_rank = np.sum(s_vals > tol)
        
        if effective_rank < rank:
            # Dependencies exist.
            # This is a complex reduction. 
            # For now, if we detect dependencies, we might try to return a subset.
            # However, identifying the correct subset that sums to T_mm is non-trivial.
            # If we just drop columns, the sum changes.
            # Given the risk, and that Strassen is likely optimal, we stick to rank 189.
            pass
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
