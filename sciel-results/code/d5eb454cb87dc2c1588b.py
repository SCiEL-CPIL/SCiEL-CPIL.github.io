"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 62 decomposition using Strassen for the 2x2x2 core block.
    # The 2x2x2 block corresponds to i in {0,1}, j in {0,1}, k in {0,1}.
    # This replaces 8 schoolbook terms with 7 Strassen terms.
    # The remaining terms are:
    # 1. i in {0,1}, j=2, k in {0..6} -> 2*7 = 14 terms
    # 2. i=2, j in {0,1,2}, k in {0..6} -> 3*7 = 21 terms
    # 3. i in {0,1}, j in {0,1}, k in {2..6} -> 2*2*5 = 20 terms
    # Total remaining: 14 + 21 + 20 = 55.
    # Total rank: 7 + 55 = 62.
    
    rank = 62
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeff=1):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += coeff
        for idx in v_indices:
            V[idx][col] += coeff
        for idx in w_indices:
            W[idx][col] += coeff
        col += 1

    # Strassen 2x2x2 decomposition for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 00->0, 01->1, 10->3, 11->4 (since m=3, index = i*m+j)
    # B indices: 00->0, 01->1, 10->7, 11->8 (since p=7, index = j*p+k)
    # C indices: 00->0, 01->1, 10->7, 11->8 (since n=3, index = k*n+i? No, W[k*n+i]. 
    # Wait, W index is k*n+i.
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 3
    # C[1,0] -> k=0, i=1 -> 1
    # C[1,1] -> k=1, i=1 -> 4
    
    # M1 = (A00+A11)(B00+B11)
    add_term([0, 4], [0, 8], [0, 4]) # C00, C11
    
    # M2 = (A10+A11)B00
    add_term([3, 4], [0], [1, 4]) # C10, C11
    
    # M3 = A00(B10-B11)
    add_term([0], [7, 8], [3, 4]) # C01, C11 with coeff -1 for B11?
    # Wait, M3 = A00(B10-B11). 
    # This is A00*B10 - A00*B11.
    # My add_term adds coeff to all.
    # I need to handle signs.
    # Let's modify add_term to take a list of (idx, coeff) or just handle it manually.
    
    # Let's restart the term addition with explicit signs.
    col = 0
    def add_term_signed(u_pairs, v_pairs, w_pairs):
        nonlocal col
        for idx, c in u_pairs:
            U[idx][col] += c
        for idx, c in v_pairs:
            V[idx][col] += c
        for idx, c in w_pairs:
            W[idx][col] += c
        col += 1

    # M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    add_term_signed([(0,1), (4,1)], [(0,1), (8,1)], [(0,1), (4,1)])
    
    # M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    add_term_signed([(3,1), (4,1)], [(0,1)], [(1,1), (4,-1)])
    
    # M3 = A00(B10-B11)
    # C01 += M3, C11 += M3
    add_term_signed([(0,1)], [(7,1), (8,-1)], [(3,1), (4,1)])
    
    # M4 = A11(B01-B00)
    # C00 += M4, C10 += M4
    add_term_signed([(4,1)], [(1,1), (0,-1)], [(0,1), (1,1)])
    
    # M5 = (A00+A01)B11
    # C00 -= M5, C01 += M5
    add_term_signed([(0,1), (1,1)], [(8,1)], [(0,-1), (3,1)])
    
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    add_term_signed([(3,1), (0,-1)], [(0,1), (1,1)], [(4,1)])
    
    # M7 = (A01-A11)(B10+B11)
    # C00 += M7
    add_term_signed([(1,1), (4,-1)], [(7,1), (8,1)], [(0,1)])
    
    # Now add the 55 schoolbook terms
    # 1. i in {0,1}, j=2, k in {0..6}
    for i in [0, 1]:
        for k in range(7):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            add_term_signed([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
            
    # 2. i=2, j in {0,1,2}, k in {0..6}
    for j in range(3):
        for k in range(7):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            add_term_signed([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
            
    # 3. i in {0,1}, j in {0,1}, k in {2..6}
    for i in [0, 1]:
        for j in [0, 1]:
            for k in range(2, 7):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_term_signed([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
