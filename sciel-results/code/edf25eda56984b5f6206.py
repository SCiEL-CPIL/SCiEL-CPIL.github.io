import numpy as np
import time

# EVOLVE-BLOCK-START

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _check_deadline():
    if time.time() - _START_TIME > TIME_LIMIT:
        raise TimeoutError("Deadline exceeded")

def _get_strassen_2x2_factors():
    """
    Returns lists of U, V, W matrices (2x2) for the 7 Strassen terms.
    U[i, k] corresponds to A[i, k]
    V[k, j] corresponds to B[k, j]
    W[j, i] corresponds to C[i, j] (Note: W is indexed by output indices)
    
    The decomposition is:
    C[i, j] = sum_r U_r[i, k] A[i, k] V_r[k, j] B[k, j] W_r[j, i]
    Wait, standard tensor contraction:
    M_{ik, kj, ji} = sum_r U_{ik, r} V_{kj, r} W_{ji, r}
    Here U, V, W are the factors.
    """
    # The 7 linear combinations
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(e)
    # P3 = (a-b)(f-h)
    # P4 = (d)(g-h)  -- Correction: P4 = (b)(d-h)? 
    # Let's use the standard explicit matrices.
    
    # Term 1
    U1 = np.array([[1, 0], [0, 1]], dtype=float)
    V1 = np.array([[1, 0], [0, 1]], dtype=float)
    W1 = np.array([[1, -1], [1, -1]], dtype=float)
    
    # Term 2
    U2 = np.array([[0, 0], [1, 1]], dtype=float)
    V2 = np.array([[1, 0], [0, 0]], dtype=float)
    W2 = np.array([[0, 0], [1, -1]], dtype=float)
    
    # Term 3
    U3 = np.array([[1, -1], [0, 0]], dtype=float)
    V3 = np.array([[0, 1], [0, -1]], dtype=float)
    W3 = np.array([[-1, 0], [0, 1]], dtype=float)
    
    # Term 4
    U4 = np.array([[0, 0], [0, 1]], dtype=float)
    V4 = np.array([[0, 0], [1, -1]], dtype=float)
    W4 = np.array([[0, 0], [0, 1]], dtype=float)
    
    # Term 5
    U5 = np.array([[1, 0], [0, 0]], dtype=float)
    V5 = np.array([[1, 0], [0, 1]], dtype=float)
    W5 = np.array([[1, 0], [0, 0]], dtype=float)
    
    # Term 6
    U6 = np.array([[0, 1], [0, 0]], dtype=float)
    V6 = np.array([[0, 1], [0, 0]], dtype=float)
    W6 = np.array([[0, 1], [0, 0]], dtype=float)
    
    # Term 7
    U7 = np.array([[0, 0], [1, 0]], dtype=float)
    V7 = np.array([[0, 0], [0, 1]], dtype=float)
    W7 = np.array([[0, 0], [1, 0]], dtype=float)

    return [U1, U2, U3, U4, U5, U6, U7], [V1, V2, V3, V4, V5, V6, V7], [W1, W2, W3, W4, W5, W6, W7]

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U2s, V2s, W2s = _get_strassen_2x2_factors()
    
    # Recursive Strassen construction
    # For 4x4, we split into 2x2 blocks.
    # The 49 terms correspond to (outer_term, inner_term)
    
    for t_outer in range(7):
        for t_inner in range(7):
            r_idx = t_outer * 7 + t_inner
            
            U_outer = U2s[t_outer]
            V_outer = V2s[t_outer]
            W_outer = W2s[t_outer]
            
            U_inner = U2s[t_inner]
            V_inner = V2s[t_inner]
            W_inner = W2s[t_inner]
            
            # The factor for the 4x4 case is the Kronecker product of the 2x2 factors?
            # No, it's a block structure.
            # A = [[A11, A12], [A21, A22]]
            # The linear form for A in the outer step is a combination of blocks.
            # U_outer selects which blocks to sum.
            # U_inner selects which elements within the block to sum.
            
            # If U_outer is, say, [[1,0],[0,1]] (identity), it means:
            # Outer term uses A11 and A22 (diagonal blocks).
            # Actually, U_outer[i, k] indicates the coefficient for block A_{ik}.
            # U_inner[i, k] indicates the coefficient for element A_{ik} within the block.
            
            # So for a specific element A_{global_i, global_k}:
            # global_i = block_i * 2 + local_i
            # global_k = block_k * 2 + local_k
            # Coeff = U_outer[block_i, block_k] * U_inner[local_i, local_k]
            
            for bi in range(2):
                for bk in range(2):
                    coeff_outer = U_outer[bi, bk]
                    if coeff_outer == 0:
                        continue
                    for li in range(2):
                        for lk in range(2):
                            coeff_inner = U_inner[li, lk]
                            if coeff_inner == 0:
                                continue
                            
                            gi = bi * 2 + li
                            gk = bk * 2 + lk
                            idx_u = gi * m + gk
                            U[idx_u, r_idx] += coeff_outer * coeff_inner
                            
            # Same for V
            # V is indexed by (block_j, block_k)? 
            # V_outer[k, j] ? 
            # In 2x2, V[k, j] corresponds to B[k, j].
            # global_k = block_k * 2 + local_k
            # global_j = block_j * 2 + local_j
            # V index in flat: k * p + j.
            for bk in range(2):
                for bj in range(2):
                    coeff_outer = V_outer[bk, bj]
                    if coeff_outer == 0:
                        continue
                    for lk in range(2):
                        for lj in range(2):
                            coeff_inner = V_inner[lk, lj]
                            if coeff_inner == 0:
                                continue
                            
                            gk = bk * 2 + lk
                            gj = bj * 2 + lj
                            idx_v = gk * p + gj
                            V[idx_v, r_idx] += coeff_outer * coeff_inner
                            
            # Same for W
            # W is indexed by (global_j, global_i).
            # W_outer[j, i].
            for bj in range(2):
                for bi in range(2):
                    coeff_outer = W_outer[bj, bi]
                    if coeff_outer == 0:
                        continue
                    for lj in range(2):
                        for li in range(2):
                            coeff_inner = W_inner[lj, li]
                            if coeff_inner == 0:
                                continue
                            
                            gj = bj * 2 + lj
                            gi = bi * 2 + li
                            idx_w = gj * n + gi
                            W[idx_w, r_idx] += coeff_outer * coeff_inner
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
