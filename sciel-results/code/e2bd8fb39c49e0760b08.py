"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=6
    
    # We use Laderman's algorithm for 3x3x3 matrix multiplication (rank 23)
    # and apply it to two 3x3 blocks of the 3x6 matrix.
    # This gives a rank of 23 + 23 = 46.
    
    # Laderman's factors for 3x3x3
    # U: 9x23, V: 9x23, W: 9x23
    # Indices:
    # U row: i*m + j (i=0..2, j=0..2)
    # V row: j*p + k (j=0..2, k=0..2)
    # W row: k*n + i (k=0..2, i=0..2)
    
    # Laderman's 23 factors (a, b, c) corresponding to U, V, W entries
    # Format: (U_index, V_index, W_index, coeff)
    # We will build the matrices explicitly.
    
    # Standard Laderman factors
    # Let's list the 23 terms.
    # Source: Various references on rank 23 for 3x3x3.
    # One common set:
    
    # We define the factors as lists of (row, col, val) for each matrix
    # Or simply define the 23 vectors.
    
    # Vectorized construction for 3x3x3
    # U_3x3: shape (9, 23)
    # V_3x3: shape (9, 23)
    # W_3x3: shape (9, 23)
    
    # The 23 factor tuples (u, v, w) where u,v,w are indices in 0..8
    # and the coefficient is 1 (or -1).
    # Actually, it's easier to hardcode the matrices from a known source.
    
    # Here is a known rank 23 decomposition for 3x3x3.
    # U (9x23), V (9x23), W (9x23)
    
    # Let's use a Python list of the 23 factors.
    # Each factor is a tuple (u_idx, v_idx, w_idx, val)
    # u_idx = i*3 + j
    # v_idx = j*3 + k
    # w_idx = k*3 + i
    
    # Laderman's factors:
    factors_3x3 = [
        (0, 0, 0, 1),   # 0*3+0, 0*3+0, 0*3+0
        (1, 3, 1, 1),   # 0*3+1, 1*3+0, 0*3+1
        (2, 6, 2, 1),   # 0*3+2, 2*3+0, 0*3+2
        (3, 1, 3, 1),   # 1*3+0, 0*3+1, 1*3+0
        (4, 4, 4, 1),   # 1*3+1, 1*3+1, 1*3+1
        (5, 7, 5, 1),   # 1*3+2, 2*3+1, 1*3+2
        (6, 2, 6, 1),   # 2*3+0, 0*3+2, 2*3+0
        (7, 5, 7, 1),   # 2*3+1, 1*3+2, 2*3+1
        (8, 8, 8, 1),   # 2*3+2, 2*3+2, 2*3+2
        (0, 4, 4, 1),   # 0*3+0, 1*3+1, 1*3+1
        (1, 7, 5, 1),   # 0*3+1, 2*3+1, 1*3+2
        (2, 1, 3, 1),   # 0*3+2, 0*3+1, 1*3+0
        (3, 8, 8, 1),   # 1*3+0, 2*3+2, 2*3+2
        (4, 2, 6, 1),   # 1*3+1, 0*3+2, 2*3+0
        (5, 5, 7, 1),   # 1*3+2, 1*3+2, 2*3+1
        (6, 3, 1, 1),   # 2*3+0, 1*3+0, 0*3+1
        (7, 0, 0, 1),   # 2*3+1, 0*3+0, 0*3+0
        (8, 6, 2, 1),   # 2*3+2, 2*3+0, 0*3+2
        (0, 6, 2, -1),  # 0*3+0, 2*3+0, 0*3+2
        (1, 1, 3, -1),  # 0*3+1, 0*3+1, 1*3+0
        (2, 5, 7, -1),  # 0*3+2, 1*3+2, 2*3+1
        (3, 7, 5, -1),  # 1*3+0, 2*3+1, 1*3+2
        (4, 0, 0, -1),  # 1*3+1, 0*3+0, 0*3+0
    ]
    
    # Wait, the above list might not be perfectly correct or complete for rank 23.
    # Let's verify the count. There are 23 entries.
    # However, constructing this from memory is risky.
    
    # A safer approach: Use the schoolbook decomposition (rank 54) which is guaranteed correct.
    # Then, if we have time, we can try to reduce it.
    # But rank 54 is a valid integer solution.
    
    # Let's stick to Schoolbook for guaranteed correctness and integer types.
    # Rank = 3 * 3 * 6 = 54.
    
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
