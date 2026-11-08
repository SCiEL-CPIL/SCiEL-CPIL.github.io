"""Matrix multiplication 2x3x10 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x10 matrix using block decomposition.
    
    Strategy: Split p=10 into 5 blocks of size 2.
    Each block computes C_block = A * B_block where A is 2x3 and B_block is 3x2.
    The tensor M_{2,3,2} has rank at most 11 (7 from Strassen-like + 4 outer).
    Total rank = 5 * 11 = 55.
    
    W follows the convention W[k*n + i, r] where k is the output row index
    (0..p-1) and i is the output column index (0..n-1).
    """
    deadline = time.time() + 130.0
    n, m, p = DIMENSIONS
    
    # Define the 11 terms for M_{2,3,2}
    # Each term is a tuple: (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # u_indices: list of (i, j) pairs, u_vals: corresponding values
    # v_indices: list of (j, k) pairs, v_vals: corresponding values  
    # w_indices: list of (k, i) pairs, w_vals: corresponding values
    terms = [
        # Strassen M1: (A00+A11)(B00+B11)
        ([(0,0),(1,1)], [1,1], [(0,0),(1,1)], [1,1], [(0,0),(1,1)], [1,1]),
        # Strassen M2: (A10+A11)B00
        ([(1,0),(1,1)], [1,1], [(0,0)], [1], [(0,1),(1,1)], [1,-1]),
        # Strassen M3: A00(B01-B11)
        ([(0,0)], [1], [(0,1),(1,1)], [1,-1], [(1,0),(1,1)], [1,1]),
        # Strassen M4: A11(B10-B00)
        ([(1,1)], [1], [(1,0),(0,0)], [1,-1], [(0,0),(0,1)], [1,1]),
        # Strassen M5: (A00+A01)B11
        ([(0,0),(0,1)], [1,1], [(1,1)], [1], [(0,0),(1,0)], [-1,1]),
        # Strassen M6: (A10-A00)(B00+B01)
        ([(1,0),(0,0)], [1,-1], [(0,0),(0,1)], [1,1], [(1,1)], [1]),
        # Strassen M7: (A01-A11)(B10+B11)
        ([(0,1),(1,1)], [1,-1], [(1,0),(1,1)], [1,1], [(0,0)], [1]),
        # Outer product term 1: A02 * B20
        ([(0,2)], [1], [(2,0)], [1], [(0,0)], [1]),
        # Outer product term 2: A02 * B21
        ([(0,2)], [1], [(2,1)], [1], [(1,0)], [1]),
        # Outer product term 3: A12 * B20
        ([(1,2)], [1], [(2,0)], [1], [(0,1)], [1]),
        # Outer product term 4: A12 * B21
        ([(1,2)], [1], [(2,1)], [1], [(1,1)], [1]),
    ]
    
    num_blocks = p // 2  # 5 blocks
    rank_per_block = len(terms)  # 11
    total_rank = num_blocks * rank_per_block  # 55
    
    U = [[0] * total_rank for _ in range(n * m)]  # (n*m) x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # (m*p) x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # (p*n) x total_rank
    
    col = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        for (u_idx, u_val, v_idx, v_val, w_idx, w_val) in terms:
            for (i, j), val in zip(u_idx, u_val):
                U[i * m + j][col] += val
            for (j, k), val in zip(v_idx, v_val):
                V[j * p + b * 2 + k][col] += val
            for (k, i), val in zip(w_idx, w_val):
                W[(b * 2 + k) * n + i][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
