"""
Optimized tensor decomposition for 2x4 * 4x5 matrix multiplication.
Uses Block-Strassen decomposition.
Rank: 36
"""

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-36 tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    The matrix A (2x4) is split into two 2x2 blocks A_L and A_R.
    The matrix B (4x5) is split into two 2x5 blocks B_T and B_B.
    C = A_L B_T + A_R B_B.
    
    Each 2x2 * 2x5 product is decomposed using:
    - Strassen for the first 4 columns (2x2 * 2x2 + 2x2 * 2x2 -> rank 7 + 7 = 14)
    - Schoolbook for the last column (2x2 * 2x1 -> rank 4)
    Total per block: 18.
    Total rank: 18 + 18 = 36.
    """
    import time
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Rank-1 terms: list of (u_dict, v_dict, w_dict)
    # u_dict: {(i, j): coeff} for A[i,j]
    # v_dict: {(j, k): coeff} for B[j,k]
    # w_dict: {(i, k): coeff} for C[i,k]
    
    rank_terms = []
    
    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        """
        Adds Strassen terms for X(2x2) * Y(2x2).
        X = A[i0:i0+2, j0:j0+2]
        Y = B[j0:j0+2, k0:k0+2]
        Contributes to C[i0:i0+2, k0:k0+2]
        """
        # Standard Strassen 7 terms
        terms = [
            # m1 = (x11+x22)*(y11+y22) -> z11 += m1, z22 += m1
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2 = (x21+x22)*y11 -> z21 += m2, z22 -= m2
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3 = x11*(y12-y22) -> z12 += m3, z22 += m3
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4 = x22*(y21-y11) -> z11 += m4, z21 += m4
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5 = (x11+x12)*y22 -> z11 -= m5, z12 += m5
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6 = (x21-x11)*(y11+y12) -> z22 += m6
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            # m7 = (x12-x22)*(y21+y22) -> z11 += m7
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
            
    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        """
        Adds schoolbook terms for X(2x2) * Y(2x1).
        X = A[i0:i0+2, j0:j0+2]
        Y = B[j0:j0+2, k0]
        Contributes to C[i0:i0+2, k0]
        """
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
    
    # Block 1: A_L (cols 0-1) * B_T (rows 0-1)
    # B_T has 5 columns: [0,1], [2,3], [4]
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)   # A_L * B_T[:, 0:2]
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)   # A_L * B_T[:, 2:4]
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1) # A_L * B_T[:, 4]
    
    # Block 2: A_R (cols 2-3) * B_B (rows 2-3)
    # B_B has 5 columns: [0,1], [2,3], [4]
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)   # A_R * B_B[:, 0:2]
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)   # A_R * B_B[:, 2:4]
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1) # A_R * B_B[:, 4]
    
    rank = len(rank_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]  # (8, rank)
    V = [[0] * rank for _ in range(m * p)]  # (20, rank)
    W = [[0] * rank for _ in range(p * n)]  # (10, rank)
    
    for c, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][c] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][c] = coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
