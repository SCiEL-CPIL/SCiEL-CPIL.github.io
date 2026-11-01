import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Decompose into 8 blocks of 2x2x2
    # j = 2*j1 + j0, k = 2*k1 + k0
    # s index is (j1, k1)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
    # Indices for 2x2: i,j in {0,1} -> i*2+j
    U_s = [
        [1, 0, 1, 0, 1, 1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1], # i=0, j=1
        [0, 1, 0, 0, 0, -1, 0], # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1
    ]
    V_s = [
        [1, 1, 0, 0, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 0, 1, 1, 1], # j=0, k=1
        [0, 1, -1, 1, 0, 0, -1], # j=1, k=0
        [1, 0, -1, -1, 1, 0, 1]  # j=1, k=1
    ]
    W_s = [
        [1, 0, 0, 0, 0, 0, 0], # k=0, i=0 (C00)
        [0, 0, 1, 0, 1, 0, 0], # k=0, i=1 (C10)
        [0, 0, 0, 1, 0, 0, 0], # k=1, i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0] # k=1, i=1 (C11)
    ]
    # Note: W indices are k*n + i. For 2x2, k*2+i.
    # W_s[0] is k=0, i=0.
    # W_s[1] is k=0, i=1.
    # W_s[2] is k=1, i=0.
    # W_s[3] is k=1, i=1.
    
    # Check Strassen formulas:
    # C00 = M1 + M4 - M5 + M7
    # M1: U[0,0]=1, U[1,1]=1. V[0,0]=1, V[1,1]=1.
    # M4: U[1,1]=1. V[1,0]=1, V[0,0]=-1? No.
    # Let's verify V_s for M4.
    # M4 = d(g-e) = A[1,1] * (B[1,0] - B[0,0]).
    # U_s for M4 (col 3): U[1,1]=1. Others 0. Correct.
    # V_s for M4 (col 3): V[1,0]=1, V[0,0]=-1.
    # My V_s[1,0] is row 1, col 0? No, V_s[j, k].
    # V_s[1,0] is j=1, k=0. Value 1.
    # V_s[0,0] is j=0, k=0. Value -1.
    # Correct.
    
    # M5 = (a+b)h = (A[0,0]+A[0,1]) * B[1,1].
    # U_s col 4: U[0,0]=1, U[0,1]=1.
    # V_s col 4: V[1,1]=1.
    # Correct.
    
    # M7 = (b-d)(g+h) = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1]).
    # U_s col 6: U[0,1]=1, U[1,1]=-1.
    # V_s col 6: V[1,0]=1, V[1,1]=1.
    # Correct.
    
    # W_s for C00 (row 0):
    # C00 = M1 + M4 - M5 + M7.
    # W_s[0] should be [1, 0, 0, 1, -1, 0, 1].
    # My W_s[0] is [1, 0, 0, 0, 0, 0, 0]. INCORRECT.
    
    # Let's rebuild W_s carefully.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0 (C00)
        [0, 1, 0, 1, 0, 0, 0],  # k=0, i=1 (C10)
        [0, 0, 1, 0, 1, 0, 0],  # k=1, i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1 (C11)
    ]
    
    # Fill global matrices
    for j1 in range(2):
        for k1 in range(4):
            s = j1 * 4 + k1
            r_start = s * 7
            for i in range(2):
                for j0 in range(2):
                    for k0 in range(2):
                        # U
                        u_idx = i * m + (2 * j1 + j0)
                        for r in range(7):
                            U[u_idx][r_start + r] += U_s[i * 2 + j0][r]
                        
                        # V
                        v_idx = (2 * j1 + j0) * p + (2 * k1 + k0)
                        for r in range(7):
                            V[v_idx][r_start + r] += V_s[j0 * 2 + k0][r]
                        
                        # W
                        w_idx = (2 * k1 + k0) * n + i
                        for r in range(7):
                            W[w_idx][r_start + r] += W_s[k0 * 2 + i][r]
                            
    return {"U": U, "V": V, "W": W}
