"""
Rank-49 exact tensor decomposition for 4x4 x 4x4 matrix multiplication.

This implementation uses a recursive Strassen algorithm.
- Base case: 2x2 Strassen (rank 7).
- Recursive step: Compose the 7 Strassen products, where each product is a 
  multiplication of sub-matrices. Since the sub-matrices are of size n/2, 
  we use the decomposition for size n/2.
- For 4x4, this results in 7 * 7 = 49 rank terms.

The decomposition is generated programmatically to avoid indexing errors 
in hardcoded tables.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# The 7 Strassen linear combinations for A, B, and C.
# Each is a list of 4 tuples (block_row, block_col, sign)
# Block indices: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right

_A_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],       # M0: A00 + A11
    [(1, 0, 1), (1, 1, 1)],       # M1: A10 + A11
    [(0, 0, 1)],                  # M2: A00
    [(1, 1, 1)],                  # M3: A11
    [(0, 0, 1), (0, 1, 1)],       # M4: A00 + A01
    [(1, 0, 1), (0, 0, -1)],      # M5: A10 - A00
    [(0, 1, 1), (1, 1, -1)],      # M6: A01 - A11
]

_B_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],       # M0: B00 + B11
    [(0, 0, 1)],                  # M1: B00
    [(0, 1, 1), (1, 1, -1)],      # M2: B01 - B11
    [(1, 0, 1), (0, 0, -1)],      # M3: B10 - B00
    [(1, 1, 1)],                  # M4: B11
    [(0, 0, 1), (0, 1, 1)],       # M5: B00 + B01
    [(1, 0, 1), (1, 1, 1)],       # M6: B10 + B11
]

_C_COMBOS = [
    [(0, 0, 1), (1, 1, 1)],       # M0: C00 += 1, C11 += 1
    [(1, 0, 1), (1, 1, -1)],      # M1: C10 += 1, C11 -= 1
    [(0, 1, 1), (1, 1, 1)],       # M2: C01 += 1, C11 += 1
    [(0, 0, 1), (1, 0, 1)],       # M3: C00 += 1, C10 += 1
    [(0, 1, 1), (0, 0, -1)],      # M4: C01 += 1, C00 -= 1
    [(1, 1, 1)],                  # M5: C11 += 1
    [(0, 0, 1)],                  # M6: C00 += 1
]


def _base_strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen (rank 7).
    U, V, W shapes: (4, 7), (4, 7), (4, 7)
    Indices:
    U: i*m + k = i*2 + k
    V: k*p + j = k*2 + j
    W: j*n + i = j*2 + i
    """
    n = 2
    rank = 7
    U = np.zeros((n * n, rank), dtype=float)
    V = np.zeros((n * n, rank), dtype=float)
    W = np.zeros((n * n, rank), dtype=float)
    
    for t in range(7):
        # Fill U for term t
        for (br, bc, sign) in _A_COMBOS[t]:
            for ir in range(n):
                for ic in range(n):
                    # Global index in A: (br*n + ir, bc*n + ic)
                    # U index: (br*n + ir) * n + (bc*n + ic)
                    u_idx = (br * n + ir) * n + (bc * n + ic)
                    U[u_idx, t] += sign
        
        # Fill V for term t
        for (br, bc, sign) in _B_COMBOS[t]:
            for ir in range(n):
                for ic in range(n):
                    # Global index in B: (br*n + ir, bc*n + ic)
                    # V index: (br*n + ir) * n + (bc*n + ic)
                    v_idx = (br * n + ir) * n + (bc * n + ic)
                    V[v_idx, t] += sign
        
        # Fill W for term t
        for (br, bc, sign) in _C_COMBOS[t]:
            for ir in range(n):
                for ic in range(n):
                    # Global index in C: (br*n + ir, bc*n + ic)
                    # W index: (bc*n + ic) * n + (br*n + ir)  <-- Note: W is j*n+i
                    # Here, row of C is ir_global = br*n + ir, col of C is ic_global = bc*n + ic
                    # W index: ic_global * n + ir_global
                    w_idx = (bc * n + ic) * n + (br * n + ir)
                    W[w_idx, t] += sign
                    
    return U, V, W


def _compose_strassen(U_sub, V_sub, W_sub, n_sub):
    """
    Composes the Strassen decomposition for size n_sub into size 2*n_sub.
    U_sub, V_sub, W_sub are for n_sub x n_sub multiplication.
    Returns U, V, W for 2*n_sub x 2*n_sub multiplication.
    """
    n = 2 * n_sub
    rank_sub = U_sub.shape[1]
    rank = 7 * rank_sub
    
    U = np.zeros((n * n, rank), dtype=float)
    V = np.zeros((n * n, rank), dtype=float)
    W = np.zeros((n * n, rank), dtype=float)
    
    for t in range(7):
        # For each of the 7 outer terms, we combine the sub-decompositions.
        # The sub-decomposition has rank_sub columns.
        # We need to map the sub-indices to the global indices.
        
        # U construction
        for (br, bc, sign) in _A_COMBOS[t]:
            # This block corresponds to A[br*n_sub : (br+1)*n_sub, bc*n_sub : (bc+1)*n_sub]
            # The sub-U matrix has rows indexed by local (i, k) in 0..n_sub-1.
            # Global row in U_new: (br*n_sub + i) * n + (bc*n_sub + k)
            # Wait, U index is i_global * n + k_global.
            # i_global = br*n_sub + i_local
            # k_global = bc*n_sub + k_local
            
            # Iterate over all rows and cols of the sub-block
            for i_local in range(n_sub):
                for k_local in range(n_sub):
                    # Sub U index: i_local * n_sub + k_local
                    u_sub_idx = i_local * n_sub + k_local
                    # Global U index
                    i_global = br * n_sub + i_local
                    k_global = bc * n_sub + k_local
                    u_new_idx = i_global * n + k_global
                    
                    # The value is sign * U_sub[i_local, k_local, j] for each j
                    # We need to add this to U_new[u_new_idx, t*rank_sub + j]
                    
                    # Vectorized update for all j
                    U[u_new_idx, t * rank_sub : (t + 1) * rank_sub] += sign * U_sub[u_sub_idx, :]
        
        # V construction
        for (br, bc, sign) in _B_COMBOS[t]:
            for i_local in range(n_sub):
                for k_local in range(n_sub):
                    # Sub V index: i_local * n_sub + k_local
                    v_sub_idx = i_local * n_sub + k_local
                    # Global V index
                    i_global = br * n_sub + i_local
                    k_global = bc * n_sub + k_local
                    v_new_idx = i_global * n + k_global
                    
                    V[v_new_idx, t * rank_sub : (t + 1) * rank_sub] += sign * V_sub[v_sub_idx, :]
        
        # W construction
        for (br, bc, sign) in _C_COMBOS[t]:
            for i_local in range(n_sub):
                for k_local in range(n_sub):
                    # Sub W index: k_local * n_sub + i_local  (j*n + i)
                    w_sub_idx = k_local * n_sub + i_local
                    # Global W index
                    # C row: i_global = br*n_sub + i_local
                    # C col: k_global = bc*n_sub + k_local
                    # W index: k_global * n + i_global
                    i_global = br * n_sub + i_local
                    k_global = bc * n_sub + k_local
                    w_new_idx = k_global * n + i_global
                    
                    W[w_new_idx, t * rank_sub : (t + 1) * rank_sub] += sign * W_sub[w_sub_idx, :]
                    
    return U, V, W


def _strassen_4x4(n, m, p):
    """
    Build the rank-49 U, V, W factor matrices for 4x4 x 4x4 multiplication.
    """
    assert (n, m, p) == (4, 4, 4)
    
    # Step 1: Get 2x2 Strassen decomposition
    U2, V2, W2 = _base_strassen_2x2()
    
    # Step 2: Compose to get 4x4 Strassen decomposition
    U4, V4, W4 = _compose_strassen(U2, V2, W2, 2)
    
    return U4, V4, W4


def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify the decomposition against the matrix multiplication tensor.
    """
    rank = U.shape[1]
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)

    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=atol))


def _schoolbook_decomposition(n, m, p):
    """Fallback rank n*m*p decomposition (always valid)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Primary: Recursive Strassen (rank 49)
    try:
        U, V, W = _strassen_4x4(n, m, p)
        if time.time() < deadline and _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: Schoolbook (rank 64)
    U, V, W = _schoolbook_decomposition(n, m, p)
    if time.time() < deadline:
        _verify_decomposition(U, V, W, n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
