"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_decomp():
    """Return U, V, W for 2x2 x 2x2 matrix multiplication with rank 7 (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 multiplications for 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Indices: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i (trace convention)
    # Wait, the trace convention is W[k*n+i, r]. For n=2, W[k*2+i, r].
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: A[0,0] and A[1,1] -> indices 0, 3
    # V: B[0,0] and B[1,1] -> indices 0, 3
    # W: contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # C11 = M1+..., C12 = ..., C21 = ..., C22 = M1-...
    # W[k*2+i]: C[0,0]=W[0], C[1,0]=W[2], C[0,1]=W[1], C[1,1]=W[3]
    # M1 contributes: C11 += 1, C12 += 0, C21 += 0, C22 += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C[0,0] and C[1,1]
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 = M2+... so W[2][1] = 1 (C[1,0])
    W[2][1] = 1
    # C22 = M1-M2+... so W[3][1] = -1
    W[3][1] = -1
    
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C12 = M3+... so W[1][2] = 1 (C[0,1])
    # C22 = M1-M2+M3+... so W[3][2] = 1
    W[1][2] = 1
    W[3][2] = 1
    
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C11 = M1+M4-... so W[0][3] = 1
    # C21 = M2+M4 so W[2][3] = 1
    W[0][3] = 1
    W[2][3] = 1
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    # C11 = M1+M4-M5+... so W[0][4] = -1
    # C12 = M3+M5 so W[1][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 = M1-M2+M3+M6 so W[3][5] = 1
    W[3][5] = 1
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 = M1+M4-M5+M7 so W[0][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_4x4_decomp():
    """Return U, V, W for 4x4 x 4x4 with rank 49 using recursive Strassen.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2_decomp()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # 4x4 = 2x(2x2) blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # Using Strassen at top level: 7 multiplications of 2x2 x 2x2
    # Each 2x2 x 2x2 uses the rank-7 decomposition
    
    # The 7 top-level multiplications correspond to:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # For each top-level multiplication t (0..6), we use the 2x2 Strassen decomp
    # The 2x2 decomp has 7 sub-multiplications
    
    # A block indices: A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3
    #               A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    # In flat index (i*4+j): A11: 0,1,4,5; A12: 2,3,6,7; A21: 8,9,12,13; A22: 10,11,14,15
    
    # For the 2x2 sub-problem, local indices:
    # A_local[0,0]=A[0,0], A_local[0,1]=A[0,1], A_local[1,0]=A[1,0], A_local[1,1]=A[1,1]
    # Flat: 0,1,2,3
    
    # Top-level multiplication t uses specific blocks of A and B
    # Let's define for each t, which A-blocks and B-blocks are used, with signs
    
    # t=0: M1 = (A11+A22)*(B11+B22)
    #   A part: A11 with +1, A22 with +1
    #   B part: B11 with +1, B22 with +1
    # t=1: M2 = (A21+A22)*B11
    #   A part: A21 with +1, A22 with +1
    #   B part: B11 with +1
    # t=2: M3 = A11*(B12-B22)
    #   A part: A11 with +1
    #   B part: B12 with +1, B22 with -1
    # t=3: M4 = A22*(B21-B11)
    #   A part: A22 with +1
    #   B part: B21 with +1, B11 with -1
    # t=4: M5 = (A11+A12)*B22
    #   A part: A11 with +1, A12 with +1
    #   B part: B22 with +1
    # t=5: M6 = (A21-A11)*(B11+B12)
    #   A part: A21 with +1, A11 with -1
    #   B part: B11 with +1, B12 with +1
    # t=6: M7 = (A12-A22)*(B21+B22)
    #   A part: A12 with +1, A22 with -1
    #   B part: B21 with +1, B22 with +1
    
    # A block flat indices (for 2x2 block in 4x4):
    # A11: [0, 1, 4, 5]  (rows 0,1; cols 0,1)
    # A12: [2, 3, 6, 7]  (rows 0,1; cols 2,3)
    # A21: [8, 9, 12, 13] (rows 2,3; cols 0,1)
    # A22: [10, 11, 14, 15] (rows 2,3; cols 2,3)
    
    # B block flat indices (for 2x2 block in 4x4, B is 4x4 so j*4+k):
    # B11: [0, 1, 4, 5]  (rows 0,1; cols 0,1)
    # B12: [2, 3, 6, 7]  (rows 0,1; cols 2,3)
    # B21: [8, 9, 12, 13] (rows 2,3; cols 0,1)
    # B22: [10, 11, 14, 15] (rows 2,3; cols 2,3)
    
    A_blocks = {
        'A11': [0, 1, 4, 5],
        'A12': [2, 3, 6, 7],
        'A21': [8, 9, 12, 13],
        'A22': [10, 11, 14, 15]
    }
    B_blocks = {
        'B11': [0, 1, 4, 5],
        'B12': [2, 3, 6, 7],
        'B21': [8, 9, 12, 13],
        'B22': [10, 11, 14, 15]
    }
    
    # For each top-level t, and for each sub-multiplication s (0..6) in the 2x2 Strassen:
    # The sub-problem multiplies A_sub (2x2) by B_sub (2x2)
    # A_sub is formed from the A-blocks with signs
    # B_sub is formed from the B-blocks with signs
    
    # The 2x2 Strassen U2, V2, W2 operate on local 2x2 indices
    # Local A index (i*2+j) maps to global based on which block
    
    # For a given top-level t, the A-subproblem uses certain A-blocks
    # Let's define: for top-level t, A_contrib = list of (block_name, sign)
    # Similarly B_contrib
    
    top_A = [
        [('A11', 1), ('A22', 1)],      # t=0
        [('A21', 1), ('A22', 1)],      # t=1
        [('A11', 1)],                   # t=2
        [('A22', 1)],                   # t=3
        [('A11', 1), ('A12', 1)],      # t=4
        [('A21', 1), ('A11', -1)],     # t=5
        [('A12', 1), ('A22', -1)],     # t=6
    ]
    top_B = [
        [('B11', 1), ('B22', 1)],      # t=0
        [('B11', 1)],                   # t=1
        [('B12', 1), ('B22', -1)],     # t=2
        [('B21', 1), ('B11', -1)],     # t=3
        [('B22', 1)],                   # t=4
        [('B11', 1), ('B12', 1)],      # t=5
        [('B21', 1), ('B22', 1)],      # t=6
    ]
    
    # C output mapping for top-level Strassen:
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # C blocks: C11 (top-left 2x2), C12 (top-right 2x2), C21 (bottom-left 2x2), C22 (bottom-right 2x2)
    # C flat indices (k*4+i):
    # C11: i in {0,1}, k in {0,1} -> k*4+i: 0,1,4,5
    # C12: i in {0,1}, k in {2,3} -> k*4+i: 8,9,12,13
    # C21: i in {2,3}, k in {0,1} -> k*4+i: 2,3,6,7
    # C22: i in {2,3}, k in {2,3} -> k*4+i: 10,11,14,15
    
    C_blocks = {
        'C11': [0, 1, 4, 5],
        'C12': [8, 9, 12, 13],
        'C21': [2, 3, 6, 7],
        'C22': [10, 11, 14, 15]
    }
    
    # For each top-level t, the contribution to C blocks:
    C_contrib = {
        0: [('C11', 1), ('C22', 1)],       # M1
        1: [('C21', 1), ('C22', -1)],      # M2
        2: [('C12', 1), ('C22', 1)],       # M3
        3: [('C11', 1), ('C21', 1)],       # M4
        4: [('C11', -1), ('C12', 1)],      # M5
        5: [('C22', 1)],                    # M6
        6: [('C11', 1)],                    # M7
    }
    
    for t in range(7):
        base = t * 7
        A_parts = top_A[t]
        B_parts = top_B[t]
        
        # For the sub 2x2 problem, local index (i*2+j) for A
        # We need to map local (i,j) to global A index
        # The A-subproblem is a virtual 2x2 matrix formed from the A-blocks
        
        # For each local A element (li, lj) where li, lj in {0,1}:
        # It gets contributions from each A-block in A_parts
        # A-block 'A11' local (li,lj) -> global (li, lj) -> flat li*4+lj
        # A-block 'A12' local (li,lj) -> global (li, lj+2) -> flat li*4+lj+2
        # A-block 'A21' local (li,lj) -> global (li+2, lj) -> flat (li+2)*4+lj
        # A-block 'A22' local (li,lj) -> global (li+2, lj+2) -> flat (li+2)*4+lj+2
        
        block_offset_A = {
            'A11': (0, 0),
            'A12': (0, 2),
            'A21': (2, 0),
            'A22': (2, 2)
        }
        block_offset_B = {
            'B11': (0, 0),
            'B12': (0, 2),
            'B21': (2, 0),
            'B22': (2, 2)
        }
        
        for s in range(7):
            col = base + s
            # U2[local_a, s], V2[local_b, s], W2[local_c, s]
            for local_a in range(4):
                li, lj = local_a // 2, local_a % 2
                for block_name, sign in A_parts:
                    row_off, col_off = block_offset_A[block_name]
                    global_a = (li + row_off) * 4 + (lj + col_off)
                    U[global_a][col] += sign * U2[local_a][s]
            
            for local_b in range(4):
                lj, lk = local_b // 2, local_b % 2
                for block_name, sign in B_parts:
                    row_off, col_off = block_offset_B[block_name]
                    global_b = (lj + row_off) * 4 + (lk + col_off)
                    V[global_b][col] += sign * V2[local_b][s]
            
            for local_c in range(4):
                lk, li = local_c // 2, local_c % 2
                # local_c = lk*2 + li, where lk is col of C, li is row of C
                # C block: C11 -> (li, lk), C12 -> (li, lk+2), C21 -> (li+2, lk), C22 -> (li+2, lk+2)
                for block_name, sign in C_contrib[t]:
                    if block_name == 'C11':
                        global_c = lk * 4 + li
                    elif block_name == 'C12':
                        global_c = (lk + 2) * 4 + li
                    elif block_name == 'C21':
                        global_c = lk * 4 + (li + 2)
                    elif block_name == 'C22':
                        global_c = (lk + 2) * 4 + (li + 2)
                    else:
                        continue
                    W[global_c][col] += sign * W2[local_c][s]
    
    return U, V, W

def _schoolbook_4x4x3_decomp():
    """Return U, V, W for 4x4 x 4x3 with rank 48 (schoolbook).
    U: (16, 48), V: (12, 48), W: (12, 48)
    Convention: U[i*4+j, r], V[j*3+k, r], W[k*4+i, r]
    n=4, m=4, p=3
    """
    n, m, p = 4, 4, 3
    rank = n * m * p  # 48
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Decompose 4x4 x 4x7 into 4x4 x 4x4 (cols 0-3) and 4x4 x 4x3 (cols 4-6)
    # Wait, B is 4x7, so columns 0-6. Split into B_left (4x4, cols 0-3) and B_right (4x3, cols 4-6)
    # C = A * B = A * [B_left | B_right] = [A*B_left | A*B_right]
    # A*B_left is 4x4, A*B_right is 4x3
    # C is 4x7, so C_left (4x4, cols 0-3) and C_right (4x3, cols 4-6)
    
    # For the 4x4 x 4x4 part:
    U1, V1, W1 = _strassen_4x4_decomp()
    # U1: (16, 49), V1: (16, 49), W1: (16, 49)
    # But W1 is for 4x4 output with trace convention W[k*4+i, r]
    # For the full 4x7 problem, W should be (7*4, rank) = (28, rank)
    # The 4x4 part contributes to W[k*4+i, r] for k in {0,1,2,3} (first 4 columns of C)
    
    # For the 4x4 x 4x3 part:
    U2, V2, W2 = _schoolbook_4x4x3_decomp()
    # U2: (16, 48), V2: (12, 48), W2: (12, 48)
    # W2 is for 4x3 output with trace convention W[k*4+i, r] for k in {0,1,2} (but these are local k)
    # For the full problem, these correspond to k in {4,5,6} (last 3 columns of C)
    
    total_rank = 49 + 48  # 97
    
    U = [[0]*total_rank for _ in range(16)]  # n*m = 4*4 = 16
    V = [[0]*total_rank for _ in range(28)]  # m*p = 4*7 = 28
    W = [[0]*total_rank for _ in range(28)]  # p*n = 7*4 = 28
    
    # Copy part 1 (4x4 x 4x4)
    for i in range(16):
        for r in range(49):
            U[i][r] = U1[i][r]
    for i in range(16):
        for r in range(49):
            V[i][r] = V1[i][r]  # V1 is for B_left (4x4), indices 0-15
    for i in range(16):
        for r in range(49):
            W[i][r] = W1[i][r]  # W1 is for C_left (4x4), indices 0-15
    
    # Copy part 2 (4x4 x 4x3)
    offset = 49
    for i in range(16):
        for r in range(48):
            U[i][offset + r] = U2[i][r]
    for i in range(12):
        for r in range(48):
            V[16 + i][offset + r] = V2[i][r]  # B_right starts at index 16 in V
    for i in range(12):
        for r in range(48):
            W[16 + i][offset + r] = W2[i][r]  # C_right starts at index 16 in W
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

