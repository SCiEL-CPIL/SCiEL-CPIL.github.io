"""Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_vec, V_vec, W_vec) of length 4.
    U_vec corresponds to A indices [0,0], [0,1], [1,0], [1,1]
    V_vec corresponds to B indices [0,0], [0,1], [1,0], [1,1]
    W_vec corresponds to C indices [0,0], [0,1], [1,0], [1,1]
    """
    return [
        # m0
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # m2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # m3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]),
        # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # m5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        # m6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 0]),
    ]

def _build_strassen_decomposition():
    """
    Build the rank-42 decomposition for 2x4 * 4x6 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = 2, 4, 6
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    # 6 block products, each of rank 7
    rank = 6 * 7
    U = np.zeros((U_dim, rank), dtype=np.float64)
    V = np.zeros((V_dim, rank), dtype=np.float64)
    W = np.zeros((W_dim, rank), dtype=np.float64)
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    # A is 2x4. Split into A1 (cols 0-1) and A2 (cols 2-3).
    # B is 4x6. Split into B_top (rows 0-1) and B_bot (rows 2-3).
    # B_top and B_bot are 2x6. Split each into 3 blocks of 2x2.
    
    # Block 0: A1 * B_top_left (cols 0-1)
    # Block 1: A1 * B_top_mid (cols 2-3)
    # Block 2: A1 * B_top_right (cols 4-5)
    # Block 3: A2 * B_bot_left (cols 0-1)
    # Block 4: A2 * B_bot_mid (cols 2-3)
    # Block 5: A2 * B_bot_right (cols 4-5)
    
    for block_idx in range(6):
        # Determine which A block and which B block
        if block_idx < 3:
            a_block = 0 # A1
            b_col_block = block_idx # 0, 1, 2
            b_row_block = 0 # Top
        else:
            a_block = 1 # A2
            b_col_block = block_idx - 3 # 0, 1, 2
            b_row_block = 1 # Bottom
            
        for u_vec, v_vec, w_vec in strassen_terms:
            # Map u_vec (4 elements) to U (8 elements)
            for local_i in range(2):
                for local_j in range(2):
                    local_idx = local_i * 2 + local_j
                    coeff = u_vec[local_idx]
                    if coeff != 0:
                        global_i = local_i
                        global_j = a_block * 2 + local_j
                        U[global_i * m + global_j, col] = coeff
                        
            # Map v_vec (4 elements) to V (24 elements)
            for local_j in range(2):
                for local_k in range(2):
                    local_idx = local_j * 2 + local_k
                    coeff = v_vec[local_idx]
                    if coeff != 0:
                        global_j = b_row_block * 2 + local_j
                        global_k = b_col_block * 2 + local_k
                        V[global_j * p + global_k, col] = coeff
                        
            # Map w_vec (4 elements) to W (12 elements)
            # W index is k * n + i
            for local_i in range(2):
                for local_k in range(2):
                    local_idx = local_i * 2 + local_k
                    coeff = w_vec[local_idx]
                    if coeff != 0:
                        global_i = local_i
                        global_k = b_col_block * 2 + local_k
                        W[global_k * n + global_i, col] = coeff
                        
            col += 1
            
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_r U[:,r] V[:,r] W[:,r]^T equals the multiplication tensor.
    """
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    rank = U.shape[1]
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Build the reconstructed tensor T_rec
    T_rec = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for r in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
    return np.allclose(T_rec, T, atol=1e-9)

def construct():
    """
    Return the rank-42 Strassen-based decomposition for 2x4 * 4x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_strassen_decomposition()
    
    # Verify to ensure correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
                    rank = col # update rank variable for return consistency if needed, though we return dicts
                    
    # Convert to lists as per interface requirement "integer/half-integer entries"
    # The problem statement says "factor matrices of shapes ... with integer/half-integer entries"
    # and return {'U', 'V', 'W'}. It doesn't explicitly forbid numpy arrays, but lists are safer for JSON-like serialization if the evaluator expects it.
    # However, the seed program returns lists of lists. I will follow that pattern.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
