import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for 3x4x7
    # Split n=3 into 2+1
    # Part 1: 2x4x7
    # Part 2: 1x4x7
    
    # Part 2: 1x4x7 is schoolbook rank 28
    # Part 1: 2x4x7
    # Split m=4 into 2+2
    # Part 1a: 2x2x7
    # Part 1b: 2x2x7
    
    # Part 1a: 2x2x7
    # Split p=7 into 2+2+2+1
    # 3 blocks of 2x2x2 (Strassen, rank 7)
    # 1 block of 2x2x1 (Schoolbook, rank 2)
    # Total rank for 2x2x7: 3*7 + 2 = 23
    
    # Total rank: 2 * 23 (for 2x4x7) + 28 (for 1x4x7) = 46 + 28 = 74
    
    rank = 74
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a term
    # a_idx: list of (i, j, coeff) for A
    # b_idx: list of (j, k, coeff) for B
    # c_idx: list of (i, k, coeff) for C
    def add_term(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col
        if col >= rank:
            return
        for (i, j, c_a) in a_coeffs:
            U[i * m + j, col] += c_a
        for (j, k, c_b) in b_coeffs:
            V[j * p + k, col] += c_b
        for (i, k, c_c) in c_coeffs:
            W[k * n + i, col] += c_c
        col += 1

    # --- Part 2: 1x4x7 (Rows 2 of A, all cols of B, Row 2 of C) ---
    # A row index 2
    # B all
    # C row index 2
    for j in range(m):
        for k in range(p):
            # Term: A[2, j] * B[j, k] -> C[2, k]
            add_term(
                [(2, j, 1)],
                [(j, k, 1)],
                [(2, k, 1)]
            )
            
    # --- Part 1: 2x4x7 (Rows 0,1 of A) ---
    # Split into two 2x2x7 blocks
    # Block 1: A cols 0,1
    # Block 2: A cols 2,3
    
    for block in range(2):
        # block 0: j in {0, 1}
        # block 1: j in {2, 3}
        j_offset = block * 2
        
        # Split p=7 into 2+2+2+1
        # Blocks of k:
        # k_block 0: k in {0, 1}
        # k_block 1: k in {2, 3}
        # k_block 2: k in {4, 5}
        # k_block 3: k in {6}
        
        for k_block in range(4):
            if k_block < 3:
                # 2x2x2 Strassen
                k_offset = k_block * 2
                
                # Strassen terms for 2x2x2
                # M1 = (A00+A11)(B00+B11)
                # A indices: (0, j_offset+0), (1, j_offset+1)
                # B indices: (j_offset+0, k_offset+0), (j_offset+1, k_offset+1)
                # C indices: (0, k_offset+0), (0, k_offset+1), (1, k_offset+0), (1, k_offset+1)
                
                # We need to map the local 2x2x2 indices to global
                # Local A: (0,0), (1,1) -> Global (0, j_off), (1, j_off+1)
                # Local B: (0,0), (1,1) -> Global (j_off, k_off), (j_off+1, k_off+1)
                # Local C: (0,0)..(1,1) -> Global (0, k_off)..(1, k_off+1)
                
                # Term 1
                add_term(
                    [(0, j_offset, 1), (1, j_offset+1, 1)],
                    [(j_offset, k_offset, 1), (j_offset+1, k_offset+1, 1)],
                    [(0, k_offset, 1), (0, k_offset+1, 1), (1, k_offset, 1), (1, k_offset+1, 1)]
                )
                
                # Term 2: (A10+A11)B00
                # A: (1, j_off), (1, j_off+1)
                # B: (j_off, k_off)
                # C: (1, k_off)
                add_term(
                    [(1, j_offset, 1), (1, j_offset+1, 1)],
                    [(j_offset, k_offset, 1)],
                    [(1, k_offset, 1)]
                )
                
                # Term 3: A00(B01-B11)
                # A: (0, j_off)
                # B: (j_off+1, k_off+1) with -1, (j_off, k_off+1) with 1? 
                # Wait, B01 is B[0,1] in local, which is B[j_off, k_off+1] global.
                # B11 is B[1,1] in local, which is B[j_off+1, k_off+1] global.
                # So B coeffs: (j_off, k_off+1, 1), (j_off+1, k_off+1, -1)
                # C: (0, k_off+1)
                add_term(
                    [(0, j_offset, 1)],
                    [(j_offset, k_offset+1, 1), (j_offset+1, k_offset+1, -1)],
                    [(0, k_offset+1, 1)]
                )
                
                # Term 4: A11(B10-B00)
                # A: (1, j_off+1)
                # B: (j_off+1, k_off, 1), (j_off, k_off, -1)
                # C: (1, k_off)
                add_term(
                    [(1, j_offset+1, 1)],
                    [(j_offset+1, k_offset, 1), (j_offset, k_offset, -1)],
                    [(1, k_offset, 1)]
                )
                
                # Term 5: (A00+A01)B11
                # A: (0, j_off), (0, j_off+1)
                # B: (j_off+1, k_off+1)
                # C: (0, k_off+1), (1, k_off+1)
                add_term(
                    [(0, j_offset, 1), (0, j_offset+1, 1)],
                    [(j_offset+1, k_offset+1, 1)],
                    [(0, k_offset+1, 1), (1, k_offset+1, 1)]
                )
                
                # Term 6: (A10-A00)(B00+B01)
                # A: (1, j_off, 1), (0, j_off, -1)
                # B: (j_off, k_off, 1), (j_off, k_off+1, 1)
                # C: (1, k_off), (1, k_off+1)
                add_term(
                    [(1, j_offset, 1), (0, j_offset, -1)],
                    [(j_offset, k_offset, 1), (j_offset, k_offset+1, 1)],
                    [(1, k_offset, 1), (1, k_offset+1, 1)]
                )
                
                # Term 7: (A01-A11)(B10+B11)
                # A: (0, j_off+1, 1), (1, j_off+1, -1)
                # B: (j_off+1, k_off, 1), (j_off+1, k_offset+1, 1)
                # C: (0, k_off+1), (1, k_offset+1)
                add_term(
                    [(0, j_offset+1, 1), (1, j_offset+1, -1)],
                    [(j_offset+1, k_offset, 1), (j_offset+1, k_offset+1, 1)],
                    [(0, k_offset+1, 1), (1, k_offset+1, 1)]
                )
                
            else:
                # 2x2x1 Schoolbook
                k_offset = 6
                # Terms:
                # A00 B00 -> C00
                add_term(
                    [(0, j_offset, 1)],
                    [(j_offset, k_offset, 1)],
                    [(0, k_offset, 1)]
                )
                # A01 B10 -> C01? No, C is 2x1. C00, C10.
                # Local C: (0,0), (1,0). Global: (0, 6), (1, 6).
                # A01 is A[0, j_off+1]. B10 is B[j_off+1, k_off].
                # Product contributes to C[0, k_off]?
                # C[0, k] = A[0, j0] B[j0, k] + A[0, j1] B[j1, k].
                # So yes.
                add_term(
                    [(0, j_offset+1, 1)],
                    [(j_offset+1, k_offset, 1)],
                    [(0, k_offset, 1)]
                )
                # A10 B00 -> C10
                add_term(
                    [(1, j_offset, 1)],
                    [(j_offset, k_offset, 1)],
                    [(1, k_offset, 1)]
                )
                # A11 B10 -> C10
                add_term(
                    [(1, j_offset+1, 1)],
                    [(j_offset+1, k_offset, 1)],
                    [(1, k_offset, 1)]
                )
                
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
