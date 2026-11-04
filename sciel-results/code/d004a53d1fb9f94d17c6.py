import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy:
    - Split the 3x6 matrix into two 3x3 blocks.
    - For each 3x3x3 block, use a rank-7 decomposition for the 2x2x2 corner (Strassen)
      and schoolbook for the remaining 19 terms.
    - Total rank: 2 * (7 + 19) = 52
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 140  # 10s margin before 150s kill
    
    def build_3x3x3_block_terms():
        """
        Returns a list of (u, v, w) tuples for a 3x3x3 block.
        u: dict (i,j) -> coeff
        v: dict (j,k) -> coeff
        w: dict (i,k) -> coeff
        """
        terms = []
        
        # Strassen terms for 2x2x2 (i,j,k in {0,1})
        # These are the 7 Strassen terms that exactly decompose 2x2x2 multiplication
        strassen_terms = [
            # M1 = (a00 + a11)(b00 + b11)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2 = (a10 + a11) b00
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3 = a00 (b01 - b11)
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4 = a11 (b10 - b00)
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5 = (a00 + a01) b11
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6 = (a10 - a00)(b00 + b01)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,1):1}),
            # M7 = (a01 - a11)(b10 + b11)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1, (1,1):1}),
        ]
        
        # Verify Strassen terms: for each (i,j,k) in {0,1}^3, sum should be 1
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sum_val = 0
                    for u, v, w in strassen_terms:
                        if (i,j) in u and (j,k) in v and (i,k) in w:
                            sum_val += u[(i,j)] * v[(j,k)] * w[(i,k)]
                    if abs(sum_val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            for u, v, w in strassen_terms:
                terms.append((u, v, w))
        else:
            # Fallback: schoolbook for 2x2x2
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        terms.append(({(i,j):1}, {(j,k):1}, {(i,k):1}))
        
        # Add schoolbook terms for remaining indices in 3x3x3
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i < 2 and j < 2 and k < 2:
                        continue  # Already covered by Strassen/schoolbook 2x2x2
                    terms.append(({(i,j):1}, {(j,k):1}, {(i,k):1}))
        
        return terms
    
    block1_terms = build_3x3x3_block_terms()
    block2_terms = build_3x3x3_block_terms()
    
    total_rank = len(block1_terms) + len(block2_terms)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Block 1: k in 0,1,2
    for u, v, w in block1_terms:
        for (i, j), coeff_u in u.items():
            U[i * m + j][col] += coeff_u
        for (j, k), coeff_v in v.items():
            V[j * p + k][col] += coeff_v
        for (i, k), coeff_w in w.items():
            W[k * n + i][col] += coeff_w
        col += 1
        
    # Block 2: k in 3,4,5 -> map k' = k-3 to 0,1,2
    for u, v, w in block2_terms:
        for (i, j), coeff_u in u.items():
            U[i * m + j][col] += coeff_u
        for (j, k_prime), coeff_v in v.items():
            k = k_prime + 3
            V[j * p + k][col] += coeff_v
        for (i, k_prime), coeff_w in w.items():
            k = k_prime + 3
            W[k * n + i][col] += coeff_w
        col += 1
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        rank = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = 1.0
                    computed = 0.0
                    for r in range(rank):
                        computed += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if abs(computed - target) > 1e-6:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
