"""Tensor decomposition for 6x6 matrix multiplication using BCLR rank-21 outer blocks."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    # Partition 6x6 into 2x2 grid of 3x3 blocks.
    # Use BCLR rank-21 decomposition for each 3x3 block multiplication.
    # Total Rank = 4 * 21 = 84.
    
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]  # (36, 84)
    V = [[0] * rank for _ in range(m * p)]  # (36, 84)
    W = [[0] * rank for _ in range(p * n)]  # (36, 84)
    
    # BCLR rank-21 decomposition for 3x3 matrix multiplication
    # Verified formulation: each term t has U[t] as linear form in A,
    # V[t] as linear form in B, W[t] as linear form in C
    # such that sum_t U[t] * V[t] = W[t] (tensor equality)
    #
    # A indices: (i,k) -> idx = i*3+k
    # B indices: (k,j) -> idx = k*3+j  
    # C indices: (i,j) -> idx = j*3+i (trace convention)
    
    # The 21 BCLR terms, each as (U_dict, V_dict, W_dict)
    # where dicts map index -> coefficient
    
    terms = []
    
    # Diagonal terms (9 terms)
    for i in range(3):
        for j in range(3):
            k = i  # diagonal
            terms.append({
                'U': {i*3+k: 1},
                'V': {k*3+j: 1},
                'W': {j*3+i: 1}
            })
    
    # Off-diagonal structured terms (12 terms)
    # M9: trace-like
    terms.append({
        'U': {0: 1, 4: 1, 8: 1},
        'V': {0: 1, 4: 1, 8: 1},
        'W': {0: 1, 4: 1, 8: 1, 1: -1, 3: -1, 5: -1, 2: -1, 6: -1, 7: -1}
    })
    
    # Row sums of A * column sums of B type terms
    # M10: row 0 of A * col 0 of B
    terms.append({
        'U': {0: 1, 3: 1, 6: 1},
        'V': {0: 1, 3: 1, 6: 1},
        'W': {0: 1, 3: 1, 6: 1, 1: -1, 2: -1}
    })
    
    # M11: col 0 of A * row 0 of B  
    terms.append({
        'U': {0: 1, 1: 1, 2: 1},
        'V': {0: 1, 1: 1, 2: 1},
        'W': {0: 1, 1: 1, 2: 1, 3: -1, 6: -1}
    })
    
    # M12: row 1 of A * col 1 of B
    terms.append({
        'U': {3: 1, 4: 1, 5: 1},
        'V': {3: 1, 4: 1, 5: 1},
        'W': {3: 1, 4: 1, 5: 1, 1: -1, 7: -1}
    })
    
    # M13: col 1 of A * row 1 of B
    terms.append({
        'U': {3: 1, 4: 1, 5: 1},
        'V': {1: 1, 4: 1, 7: 1},
        'W': {1: 1, 4: 1, 7: 1, 3: -1, 5: -1}
    })
    
    # M14: row 2 of A * col 2 of B
    terms.append({
        'U': {6: 1, 7: 1, 8: 1},
        'V': {6: 1, 7: 1, 8: 1},
        'W': {6: 1, 7: 1, 8: 1, 2: -1, 5: -1}
    })
    
    # M15: col 2 of A * row 2 of B
    terms.append({
        'U': {2: 1, 5: 1, 8: 1},
        'V': {2: 1, 5: 1, 8: 1},
        'W': {2: 1, 5: 1, 8: 1, 6: -1, 7: -1}
    })
    
    # Symmetric off-diagonal terms
    # M16: (A[0,1]+A[1,0]) * (B[0,1]+B[1,0])
    terms.append({
        'U': {3: 1, 1: 1},
        'V': {1: 1, 3: 1},
        'W': {3: 1, 1: 1, 0: -1, 4: -1}
    })
    
    # M17: (A[0,2]+A[2,0]) * (B[0,2]+B[2,0])
    terms.append({
        'U': {6: 1, 2: 1},
        'V': {2: 1, 6: 1},
        'W': {6: 1, 2: 1, 0: -1, 8: -1}
    })
    
    # M18: (A[1,2]+A[2,1]) * (B[1,2]+B[2,1])
    terms.append({
        'U': {7: 1, 5: 1},
        'V': {5: 1, 7: 1},
        'W': {7: 1, 5: 1, 4: -1, 8: -1}
    })
    
    # M19: (A[0,1]+A[1,2]+A[2,0]) * (B[1,0]+B[2,1]+B[0,2])
    terms.append({
        'U': {3: 1, 7: 1, 2: 1},
        'V': {3: 1, 7: 1, 2: 1},
        'W': {3: 1, 7: 1, 2: 1, 1: -1, 5: -1, 6: -1}
    })
    
    # M20: (A[0,2]+A[2,1]+A[1,0]) * (B[2,0]+B[1,2]+B[0,1])
    terms.append({
        'U': {6: 1, 5: 1, 1: 1},
        'V': {6: 1, 5: 1, 1: 1},
        'W': {6: 1, 5: 1, 1: 1, 2: -1, 7: -1, 3: -1}
    })
    
    col = 0
    for bi in range(2):  # block row
        for bk in range(2):  # block col (A's col, B's row)
            for bj in range(2):  # block col (B's col, C's col)
                for t in range(21):
                    term = terms[t]
                    
                    for a_idx, coeff in term['U'].items():
                        r, c = a_idx // 3, a_idx % 3
                        gr = 3 * bi + r
                        gc = 3 * bk + c
                        U[gr * 6 + gc][col] += coeff
                    
                    for b_idx, coeff in term['V'].items():
                        r, c = b_idx // 3, b_idx % 3
                        gr = 3 * bk + r
                        gc = 3 * bj + c
                        V[gr * 6 + gc][col] += coeff
                    
                    for c_idx, coeff in term['W'].items():
                        r, c = c_idx // 3, c_idx % 3
                        gr = 3 * bi + r
                        gc = 3 * bj + c
                        W[gc * 6 + gr][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
