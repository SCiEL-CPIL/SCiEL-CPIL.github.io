import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS # 4, 4, 7
    # We will build U, V, W as lists of lists or numpy arrays.
    # Let's use numpy for ease, then convert to list of lists if needed?
    # The return type expects {'U': ..., 'V': ..., 'W': ...}
    # The seed uses lists. I will use lists to be safe, or numpy arrays if allowed.
    # "integer/half-integer entries".
    
    # Target Rank
    # Part 1: 4x4x4 -> 8 Strassen (56)
    # Part 2: 4x4x2 -> 4 Strassen (28)
    # Part 3: 4x4x1 -> 4 Schoolbook (16)
    # Total 100
    
    rank = 100
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # A_block: 2x2 indices in A
    # B_block: 2x2 indices in B
    # C_block: 2x2 indices in C
    def add_strassen(a_i, a_j, b_j, b_k, c_k, c_i, start_rank):
        # a_i, a_j: top-left of A block
        # b_j, b_k: top-left of B block
        # c_k, c_i: top-left of C block
        
        # Strassen 7 products
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Indices in 2x2 block:
        # A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        # B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
        # C: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
        
        # Global indices
        # A_idx(i, j) = (a_i + i) * m + (a_j + j)
        # B_idx(j, k) = (b_j + j) * p + (b_k + k)
        # C_idx(k, i) = (c_k + k) * n + (c_i + i)
        
        products = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]), # C21 += M4? No, C21 = M2+M4. M2 adds to C21.
            # Wait, C21 index is (1,0) in C block.
            # M2 adds to C21.
            # M3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]), # C12 += M3
            # M4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]), # C11 += M4, C21 += M4
            # M5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]), # C11 -= M5, C12 += M5
            # M6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]), # C22 += M6
            # M7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])  # C11 += M7
        ]
        
        # Correction: I need to be very careful with signs and indices.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's re-verify M4 in C11 and C21.
        # M4 = d(g-e).
        # C11 += M4. C21 += M4.
        
        # M5 in C11 and C12.
        # C11 -= M5. C12 += M5.
        
        # M2 in C21 and C22.
        # C21 += M2. C22 -= M2.
        
        # M3 in C12 and C22.
        # C12 += M3. C22 += M3.
        
        # M1 in C11 and C22.
        # C11 += M1. C22 += M1.
        
        # M6 in C22.
        # C22 += M6.
        
        # M7 in C11.
        # C11 += M7.

        # I will implement this by explicitly adding to U, V, W for each of the 7 ranks.
        
        for r_idx in range(7):
            r = start_rank + r_idx
            # Clear U, V, W for this rank (they are 0 initially)
            
            if r_idx == 0: # M1
                # U: a+d
                U[(a_i+0)*m + (a_j+0)][r] += 1
                U[(a_i+1)*m + (a_j+1)][r] += 1
                # V: e+h
                V[(b_j+0)*p + (b_k+0)][r] += 1
                V[(b_j+1)*p + (b_k+1)][r] += 1
                # W: C11 + C22
                W[(c_k+0)*n + (c_i+0)][r] += 1
                W[(c_k+1)*n + (c_i+1)][r] += 1
                
            elif r_idx == 1: # M2
                # U: c+d
                U[(a_i+1)*m + (a_j+0)][r] += 1
                U[(a_i+1)*m + (a_j+1)][r] += 1
                # V: e
                V[(b_j+0)*p + (b_k+0)][r] += 1
                # W: C21 - C22
                W[(c_k+1)*n + (c_i+0)][r] += 1
                W[(c_k+1)*n + (c_i+1)][r] -= 1
                
            elif r_idx == 2: # M3
                # U: a
                U[(a_i+0)*m + (a_j+0)][r] += 1
                # V: f-h
                V[(b_j+0)*p + (b_k+1)][r] += 1
                V[(b_j+1)*p + (b_k+1)][r] -= 1
                # W: C12 + C22
                W[(c_k+0)*n + (c_i+1)][r] += 1
                W[(c_k+1)*n + (c_i+1)][r] += 1
                
            elif r_idx == 3: # M4
                # U: d
                U[(a_i+1)*m + (a_j+1)][r] += 1
                # V: g-e
                V[(b_j+1)*p + (b_k+0)][r] += 1
                V[(b_j+0)*p + (b_k+0)][r] -= 1
                # W: C11 + C21
                W[(c_k+0)*n + (c_i+0)][r] += 1
                W[(c_k+1)*n + (c_i+0)][r] += 1
                
            elif r_idx == 4: # M5
                # U: a+b
                U[(a_i+0)*m + (a_j+0)][r] += 1
                U[(a_i+0)*m + (a_j+1)][r] += 1
                # V: h
                V[(b_j+1)*p + (b_k+1)][r] += 1
                # W: -C11 + C12
                W[(c_k+0)*n + (c_i+0)][r] -= 1
                W[(c_k+0)*n + (c_i+1)][r] += 1
                
            elif r_idx == 5: # M6
                # U: c-a
                U[(a_i+1)*m + (a_j+0)][r] += 1
                U[(a_i+0)*m + (a_j+0)][r] -= 1
                # V: e+f
                V[(b_j+0)*p + (b_k+0)][r] += 1
                V[(b_j+0)*p + (b_k+1)][r] += 1
                # W: C22
                W[(c_k+1)*n + (c_i+1)][r] += 1
                
            elif r_idx == 6: # M7
                # U: b-d
                U[(a_i+0)*m + (a_j+1)][r] += 1
                U[(a_i+1)*m + (a_j+1)][r] -= 1
                # V: g+h
                V[(b_j+1)*p + (b_k+0)][r] += 1
                V[(b_j+1)*p + (b_k+1)][r] += 1
                # W: C11
                W[(c_k+0)*n + (c_i+0)][r] += 1
        
    # Helper to add 2x2x1 schoolbook
    def add_221(a_i, a_j, b_j, b_k, c_k, c_i, start_rank):
        # A 2x2, B 2x1, C 2x1
        # C1 = a*e + b*g
        # C2 = c*e + d*g
        for r_idx in range(4):
            r = start_rank + r_idx
            if r_idx == 0: # a*e
                U[(a_i+0)*m + (a_j+0)][r] += 1
                V[(b_j+0)*p + (b_k+0)][r] += 1
                W[(c_k+0)*n + (c_i+0)][r] += 1
            elif r_idx == 1: # b*g
                U[(a_i+0)*m + (a_j+1)][r] += 1
                V[(b_j+1)*p + (b_k+0)][r] += 1
                W[(c_k+0)*n + (c_i+0)][r] += 1
            elif r_idx == 2: # c*e
                U[(a_i+1)*m + (a_j+0)][r] += 1
                V[(b_j+0)*p + (b_k+0)][r] += 1
                W[(c_k+1)*n + (c_i+0)][r] += 1
            elif r_idx == 3: # d*g
                U[(a_i+1)*m + (a_j+1)][r] += 1
                V[(b_j+1)*p + (b_k+0)][r] += 1
                W[(c_k+1)*n + (c_i+0)][r] += 1

    # Part 1: 4x4x4 (Cols 0-3)
    # A blocks: (0,0), (0,2), (2,0), (2,2)
    # B blocks: (0,0), (0,2), (2,0), (2,2)
    # C blocks: (0,0), (0,2), (2,0), (2,2)
    
    # C11 = A11 B11 + A12 B21
    add_strassen(0, 0, 0, 0, 0, 0, 0)   # A11 B11
    add_strassen(0, 2, 2, 0, 0, 0, 7)   # A12 B21
    
    # C12 = A11 B12 + A12 B22
    add_strassen(0, 0, 0, 2, 0, 2, 14)  # A11 B12
    add_strassen(0, 2, 2, 2, 0, 2, 21)  # A12 B22
    
    # C21 = A21 B11 + A22 B21
    add_strassen(2, 0, 0, 0, 2, 0, 28)  # A21 B11
    add_strassen(2, 2, 2, 0, 2, 0, 35)  # A22 B21
    
    # C22 = A21 B12 + A22 B22
    add_strassen(2, 0, 0, 2, 2, 2, 42)  # A21 B12
    add_strassen(2, 2, 2, 2, 2, 2, 49)  # A22 B22
    
    # Part 2: 4x4x2 (Cols 4-5)
    # B cols 4,5.
    # A blocks same.
    # B top: rows 0,1; cols 4,5. (j=0, k=4)
    # B bot: rows 2,3; cols 4,5. (j=2, k=4)
    # C top: rows 0,1; cols 4,5. (k=4, i=0)
    # C bot: rows 2,3; cols 4,5. (k=4, i=2)
    
    # C_top = A11 B_top + A12 B_bot
    add_strassen(0, 0, 0, 4, 4, 0, 56)  # A11 B_top
    add_strassen(0, 2, 2, 4, 4, 0, 63)  # A12 B_bot
    
    # C_bot = A21 B_top + A22 B_bot
    add_strassen(2, 0, 0, 4, 4, 2, 70)  # A21 B_top
    add_strassen(2, 2, 2, 4, 4, 2, 77)  # A22 B_bot
    
    # Part 3: 4x4x1 (Col 6)
    # B col 6.
    # B top: rows 0,1; col 6. (j=0, k=6)
    # B bot: rows 2,3; col 6. (j=2, k=6)
    # C top: rows 0,1; col 6. (k=6, i=0)
    # C bot: rows 2,3; col 6. (k=6, i=2)
    
    # C_top = A11 B_top + A12 B_bot
    add_221(0, 0, 0, 6, 6, 0, 84)
    add_221(0, 2, 2, 6, 6, 0, 88)
    
    # C_bot = A21 B_top + A22 B_bot
    add_221(2, 0, 0, 6, 6, 2, 92)
    add_221(2, 2, 2, 6, 6, 2, 96)
    
    return {"U": U, "V": V, "W": W}
