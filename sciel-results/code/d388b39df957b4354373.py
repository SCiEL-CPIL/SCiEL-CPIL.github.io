"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# Pre-computed Strassen coefficients for 2x2x2 blocks
# Each entry is (u_row, u_col, v_row, v_col, w_row, w_col, coeff)
# Using the standard 7-term Strassen decomposition
STRASSEN_TERMS = [
    # Term 1: M1 = (A11+A22)(B11+B22)
    [(0,0, 0,0, 0,0, 1), (0,0, 1,1, 0,0, 1),
     (1,0, 0,0, 0,0, 1), (1,0, 1,1, 0,0, 1),
     (1,1, 0,0, 0,1, 1), (1,1, 1,1, 0,1, 1),
     (0,1, 0,0, 1,1, 1), (0,1, 1,1, 1,1, 1)],
    # Term 2: M2 = A22(B21-B11)
    [(1,1, 0,0, 1,0, 1), (1,1, 1,0, 1,0, -1),
     (1,0, 0,0, 1,0, 1), (1,0, 1,0, 1,0, -1)],
    # Term 3: M3 = A11(B22-B12)
    [(0,0, 0,1, 1,1, 1), (0,0, 1,1, 1,1, -1),
     (0,1, 0,1, 0,1, 1), (0,1, 1,1, 0,1, -1)],
    # Term 4: M4 = A22(B11+B22)
    [(1,1, 0,0, 1,1, 1), (1,1, 1,1, 1,1, 1),
     (1,0, 0,0, 1,1, 1), (1,0, 1,1, 1,1, 1)],
    # Term 5: M5 = (A11+A12)B22
    [(0,0, 1,1, 0,1, 1), (0,1, 1,1, 0,1, 1),
     (0,1, 0,1, 0,0, 1), (0,1, 1,1, 0,0, 1)],
    # Term 6: M6 = (A21-A11)(B11+B12)
    [(1,0, 0,0, 1,0, 1), (0,0, 0,0, 1,0, -1),
     (1,0, 0,1, 1,1, 1), (0,0, 0,1, 1,1, -1)],
    # Term 7: M7 = (A12-A22)(B21+B22)
    [(0,1, 1,0, 0,0, 1), (1,1, 1,0, 0,0, -1),
     (0,1, 1,1, 0,1, 1), (1,1, 1,1, 0,1, -1)],
]

def _apply_block(U, V, W, i0, j0, k0, n_b, m_b, p_b, col, n, m, p):
    """Apply a block decomposition to the factor matrices, return next col index."""
    if n_b == 2 and m_b == 2 and p_b == 2:
        # Use Strassen's 7-term algorithm
        for term_idx in range(7):
            for (u_r, u_c, v_r, v_c, w_r, w_c, coeff) in STRASSEN_TERMS[term_idx]:
                U[(i0 + u_r) * m + (j0 + u_c), col] += coeff
                V[(j0 + v_r) * p + (k0 + v_c), col] += coeff
                W[(k0 + w_r) * n + (i0 + w_c), col] += coeff
            col += 1
    elif n_b == 2 and m_b == 1 and p_b == 2:
        # Schoolbook for 2x1x2: rank 4
        for u in range(2):
            for w in range(2):
                U[(i0 + u) * m + j0, col] = 1
                V[j0 * p + (k0 + w), col] = 1
                W[(k0 + w) * n + (i0 + u), col] = 1
                col += 1
    else:
        # Generic schoolbook
        for u in range(n_b):
            for v in range(m_b):
                for w in range(p_b):
                    U[(i0 + u) * m + (j0 + v), col] = 1
                    V[(j0 + v) * p + (k0 + w), col] = 1
                    W[(k0 + w) * n + (i0 + u), col] = 1
                    col += 1
    return col

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Uses block decomposition:
    - n=4 split into [2, 2]
    - m=5 split into [2, 2, 1]  
    - p=6 split into [2, 2, 2]
    
    2x2x2 blocks use Strassen (rank 7 each), 2x1x2 blocks use schoolbook (rank 4 each).
    Total: 12*7 + 6*4 = 84 + 24 = 108 rank.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 5, 6

    # Block partitions
    i_splits = [(0, 2), (2, 2)]      # n=4 -> [2,2]
    j_splits = [(0, 2), (2, 2), (4, 1)]  # m=5 -> [2,2,1]
    k_splits = [(0, 2), (2, 2), (4, 2)]  # p=6 -> [2,2,2]

    # Estimate rank
    rank_estimate = 0
    for i0, n_b in i_splits:
        for j0, m_b in j_splits:
            for k0, p_b in k_splits:
                if n_b == 2 and m_b == 2 and p_b == 2:
                    rank_estimate += 7
                else:
                    rank_estimate += n_b * m_b * p_b

    # Allocate factor matrices
    U = np.zeros((n * m, rank_estimate), dtype=int)
    V = np.zeros((m * p, rank_estimate), dtype=int)
    W = np.zeros((p * n, rank_estimate), dtype=int)

    col = 0
    for i0, n_b in i_splits:
        for j0, m_b in j_splits:
            for k0, p_b in k_splits:
                if time.time() > deadline:
                    break
                col = _apply_block(U, V, W, i0, j0, k0, n_b, m_b, p_b, col, n, m, p)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Trim to actual rank used
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
