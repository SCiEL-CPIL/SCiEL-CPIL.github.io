import time
import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Rank-minimal tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses Strassen's algorithm on 7 disjoint 2x2 blocks.
    Rank = 7 * 7 = 49.
    """
    n, m, p = 2, 2, 14
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Strassen terms for 2x2 matrix multiplication
    # Format: (a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs)
    # Indices are flattened row-major: i*m + j
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:10, 3:11
    # C: 0:00, 1:01, 2:10, 3:11
    
    # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # M2 = (A10+A11)B00 -> C10 += 1, C11 += -1
    # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
    # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
    # M5 = (A00+A01)B11 -> C00 += -1, C01 += 1
    # M6 = (A10-A00)(B00+B01) -> C11 += 1
    # M7 = (A01-A11)(B10+B11) -> C00 += 1
    
    terms = [
        # a_idx, a_val, b_idx, b_val, c_idx, c_val
        # M1
        [(0, 1), (3, 1), (0, 1), (3, 1), (0, 1), (3, 1)],
        # M2
        [(2, 1), (3, 1), (0, 1), (0, 1), (2, 1), (3, -1)],
        # M3
        [(0, 1), (0, 1), (1, 1), (3, -1), (1, 1), (3, 1)],
        # M4
        [(3, 1), (3, 1), (2, 1), (0, -1), (0, 1), (2, 1)],
        # M5
        [(0, 1), (1, 1), (3, 1), (3, 1), (0, -1), (1, 1)],
        # M6
        [(2, 1), (0, -1), (0, 1), (1, 1), (3, 1), (3, 1)],
        # M7
        [(1, 1), (3, -1), (2, 1), (3, 1), (0, 1), (0, 1)],
    ]
    
    rank = 49
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    num_blocks = p // 2
    
    for b_idx in range(num_blocks):
        base_k = 2 * b_idx
        for t_idx in range(7):
            col = b_idx * 7 + t_idx
            term = terms[t_idx]
            
            # Fill U
            # term[0], term[1] are pairs of (idx, val) for A
            # Actually, I structured term as flat list of 12 items?
            # Let's unpack:
            a1, a1v, a2, a2v, b1, b1v, b2, b2v, c1, c1v, c2, c2v = term
            
            U[a1, col] += a1v
            U[a2, col] += a2v
            
            # Fill V
            # B index 0: (0, base_k), 1: (0, base_k+1), 2: (1, base_k), 3: (1, base_k+1)
            # V row index: j*p + k
            v1_row = (b1 // 2) * p + (base_k + b1 % 2)
            v2_row = (b2 // 2) * p + (base_k + b2 % 2)
            V[v1_row, col] += b1v
            V[v2_row, col] += b2v
            
            # Fill W
            # C index 0: (0, base_k), 1: (0, base_k+1), 2: (1, base_k), 3: (1, base_k+1)
            # W row index: k*n + i
            c1_row = (c1 % 2) * n + (c1 // 2)
            c2_row = (c2 % 2) * n + (c2 // 2)
            W[c1_row, col] += c1v
            W[c2_row, col] += c2v

    # Convert to lists
    def mat_to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row] for row in mat]

    return {
        "U": mat_to_list(U),
        "V": mat_to_list(V),
        "W": mat_to_list(W)
    }
