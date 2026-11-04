import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline for safety
    
    # Rank 36 decomposition using Strassen for 2x2x2 blocks
    # Structure:
    # A (2x4) = [A1 A2] where A1, A2 are 2x2
    # B (4x5) = [B1; B2] where B1, B2 are 2x5
    # C = A1 B1 + A2 B2
    # For each term Ai Bi (2x2 * 2x5):
    #   Split Bi into [B_i1 B_i2 B_i3] where B_i1, B_i2 are 2x2, B_i3 is 2x1
    #   Ai B_i1 uses Strassen (rank 7)
    #   Ai B_i2 uses Strassen (rank 7)
    #   Ai B_i3 uses Schoolbook (rank 4)
    #   Total per Ai Bi is 18.
    #   Total for C is 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1

    # Strassen terms for 2x2x2 multiplication
    # C = A B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[p+q+r+t, q+t], [q+s, p+s+r]] ? No.
    # Standard Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+h)
    # m7 = (b-c)(e+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    strassen_terms = [
        # m1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # Wait, W indices for 2x2 output are 0,1,2,3?
        # Output C is 2x2. Indices:
        # C[0,0] -> idx 0
        # C[0,1] -> idx 1
        # C[1,0] -> idx 2
        # C[1,1] -> idx 3
        # But in the global W, the index is k*n + i.
        # For a 2x2 block, if it corresponds to rows i0, i1 and cols k0, k1 of C:
        # C[i0, k0] -> k0*n + i0
        # C[i0, k1] -> k1*n + i0
        # C[i1, k0] -> k0*n + i1
        # C[i1, k1] -> k1*n + i1
    ]
    
    # Let's define the local Strassen terms carefully.
    # Local A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A11)(B01+B11)
    # m7 = (A01-A10)(B00+B11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    strassen_local = [
        # (U coeffs, V coeffs, W coeffs)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}), # m1 contributes to C00 and C11
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}), # m2 contributes to C10 and C11
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}), # m3 contributes to C01 and C11
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}), # m4 contributes to C00 and C10
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}), # m5 contributes to C00 and C01
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}), # m6 contributes to C11
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})  # m7 contributes to C00
    ]
    
    # Schoolbook terms for 2x2x1 multiplication
    # A is 2x2, B is 2x1, C is 2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    schoolbook_221 = [
        ({0:1}, {0:1}, {0:1}), # A00 B0 -> C0
        ({1:1}, {1:1}, {0:1}), # A01 B1 -> C0
        ({2:1}, {0:1}, {1:1}), # A10 B0 -> C1
        ({3:1}, {1:1}, {1:1})  # A11 B1 -> C1
    ]
    
    # We have 2 blocks of A (A1, A2) and for each, 2 Strassen parts and 1 Schoolbook part.
    # Total 2 * (2 * 7 + 1 * 4) = 36 terms.
    
    for q in range(2): # q is 0 for A1, 1 for A2
        for s in range(3): # s is 0,1 for Strassen parts of B, 2 for Schoolbook part
            if s < 2:
                # Strassen part
                # B part is 2x2
                # B global indices for this part:
                # Rows: 2*q (0 or 2), Cols: 2*s (0 or 2) to 2*s+1
                # Wait, B is 4x5.
                # B1 is rows 0-1, B2 is rows 2-3.
                # B_q is rows 2q to 2q+1.
                # Within B_q, we split cols 0-1, 2-3, 4.
                # So for s=0, cols 0-1. For s=1, cols 2-3.
                
                for u_t, v_t, w_t in strassen_local:
                    u_global = {}
                    v_global = {}
                    w_global = {}
                    
                    # Map local A indices to global A indices
                    # Local A idx: 0=00, 1=01, 2=10, 3=11
                    # Global A idx: i*m + j
                    # For A_q (q=0 or 1), the columns are 2q and 2q+1.
                    # So local col 0 -> global col 2q, local col 1 -> global col 2q+1.
                    # Local row i -> global row i.
                    for idx, val in u_t.items():
                        li = idx // 2
                        lj = idx % 2
                        gi = li
                        gj = 2*q + lj
                        u_global[gi * m + gj] = val
                        
                    # Map local B indices to global B indices
                    # Local B idx: 0=00, 1=01, 2=10, 3=11
                    # Global B idx: j*p + k
                    # For B_q (q=0 or 1), the rows are 2q and 2q+1.
                    # For part s (s=0 or 1), the cols are 2s and 2s+1.
                    for idx, val in v_t.items():
                        lj = idx // 2
                        lk = idx % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        v_global[gj * p + gk] = val
                        
                    # Map local C indices to global C indices
                    # Local C idx: 0=00, 1=01, 2=10, 3=11
                    # Global C idx: k*n + i
                    # For this block, rows i are 0,1. Cols k are 2s, 2s+1.
                    for idx, val in w_t.items():
                        li = idx // 2
                        lk = idx % 2
                        gi = li
                        gk = 2*s + lk
                        w_global[gk * n + gi] = val
                        
                    add_term(u_global, v_global, w_global)
            else:
                # Schoolbook part for 2x2x1
                # B part is 2x1, col 4.
                for u_t, v_t, w_t in schoolbook_221:
                    u_global = {}
                    v_global = {}
                    w_global = {}
                    
                    for idx, val in u_t.items():
                        li = idx // 2
                        lj = idx % 2
                        gi = li
                        gj = 2*q + lj
                        u_global[gi * m + gj] = val
                        
                    for idx, val in v_t.items():
                        lj = idx
                        gk = 4
                        gj = 2*q + lj
                        v_global[gj * p + gk] = val
                        
                    for idx, val in w_t.items():
                        li = idx
                        gk = 4
                        gi = li
                        w_global[gk * n + gi] = val
                        
                    add_term(u_global, v_global, w_global)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
