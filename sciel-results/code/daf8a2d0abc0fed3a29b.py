import numpy as np
import time
import random

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = 145.0 # 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank-7 Strassen decomposition for 2x2
    # T_2 = sum_{b=1}^7 x_b (v) y_b (w) z_b (u) ? 
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    
    # Vectors for A (u), B (v), C (w)
    # Indices for 2x2: 0:00, 1:01, 2:10, 3:11
    # A indices: 0,1,2,3
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    strassen_u = [
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 0, 1, 1], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [-1, 0, 1, 0],# M6: A21 - A11
        [0, 1, 0, -1] # M7: A12 - A22
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1: B11 + B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12 - B22
        [0, -1, 1, 0],# M4: B21 - B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11 + B12
        [0, 0, 1, 1]  # M7: B21 + B22
    ]
    strassen_w = [
        [1, 0, 0, 1], # C11 = M1 - M4 - M5 + M7
        [0, 1, 0, 0], # C12 = M3 + M5
        [0, 0, 1, 0], # C21 = M2 + M4
        [0, 0, 0, 1]  # C22 = M1 - M2 + M6 + M7
    ]
    # Note: The W matrix in the task is indexed by [k*n+i, r].
    # For 2x2, n=m=p=2.
    # C indices: k*n + i.
    # k is col of C (0,1), i is row of C (0,1).
    # Index 0: 0*2+0=0 (C00)
    # Index 1: 0*2+1=1 (C01)
    # Index 2: 1*2+0=2 (C10)
    # Index 3: 1*2+1=3 (C11)
    # My strassen_w above is indexed by C00, C01, C10, C11?
    # M1 contributes to C00 and C11.
    # Let's re-verify the W vectors for Strassen.
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # Vector w_r has length 4.
    # w_1 (for M1): [1, 0, 0, 1] -> C00+=1, C11+=1. Correct.
    # w_2 (for M2): [0, 0, 1, -1] -> C10+=1, C11-=-1. Correct.
    # w_3 (for M3): [0, 1, 0, 0] -> C01+=1. Correct.
    # w_4 (for M4): [-1, 0, 1, 0] -> C00-=-1, C10+=1. Correct.
    # w_5 (for M5): [-1, 1, 0, 0] -> C00-=-1, C01+=1. Correct.
    # w_6 (for M6): [0, 0, 0, 1] -> C11+=1. Correct.
    # w_7 (for M7): [1, 0, 0, 1] -> C00+=1, C11+=1. Correct.
    
    strassen_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 0], # M3
        [-1, 0, 1, 0],# M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 1]  # M7
    ]

    # Now for 3x3.
    # Try to find rank-21 decomposition.
    # If fails, use rank-27 schoolbook.
    
    def get_3x3_decomposition():
        # Schoolbook rank 27
        u_school = []
        v_school = []
        w_school = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u = [0]*9
                    v = [0]*9
                    w = [0]*9
                    u[i*3+j] = 1
                    v[j*3+k] = 1
                    # W index: k*n + i. n=3.
                    w[k*3+i] = 1
                    u_school.append(u)
                    v_school.append(v)
                    w_school.append(w)
        return u_school, v_school, w_school, 27

    # Attempt rank 21
    # This is a complex optimization. Given time constraints, 
    # I will skip the numerical search for rank 21 to avoid timeout 
    # and use the guaranteed rank 27 (schoolbook) for the 3x3 part.
    # Rank 27 * 7 = 189.
    # This is a solid improvement from 216.
    
    u_3, v_3, w_3, rank_3 = get_3x3_decomposition()
    
    # Construct final U, V, W
    # Rank = rank_3 * 7
    total_rank = rank_3 * 7
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    col = 0
    for a in range(rank_3):
        for b in range(7):
            # u_a is length 9, x_b (strassen_u[b]) is length 4
            # v_a is length 9, y_b (strassen_v[b]) is length 4
            # w_a is length 9, z_b (strassen_w[b]) is length 4
            
            # Map indices
            # I = 6*r + 2*c + 2*r' + c'
            # u_a index: 3*r + c
            # x_b index: 2*r' + c'
            
            for r in range(3):
                for c in range(3):
                    idx_3 = 3*r + c
                    val_u3 = u_3[a][idx_3]
                    val_v3 = v_3[a][idx_3]
                    val_w3 = w_3[a][idx_3]
                    
                    if val_u3 == 0 and val_v3 == 0 and val_w3 == 0:
                        continue
                        
                    for rp in range(2):
                        for cp in range(2):
                            idx_2 = 2*rp + cp
                            val_u2 = strassen_u[b][idx_2]
                            val_v2 = strassen_v[b][idx_2]
                            val_w2 = strassen_w[b][idx_2]
                            
                            if val_u2 == 0 and val_v2 == 0 and val_w2 == 0:
                                continue
                            
                            # Global index
                            I = 6*r + 2*c + 2*rp + cp
                            
                            U[I][col] += val_u3 * val_u2
                            V[I][col] += val_v3 * val_v2
                            W[I][col] += val_w3 * val_w2
            col += 1
            
    return {"U": U, "V": V, "W": W}
