import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 114 decomposition
    rank = 114
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen M444 block
    def add_strassen_m444(offset_k, col_offset):
        # X_l, Y_l definitions for l=0..6
        # Blocks are 2x2. A[0,0], A[0,1], A[1,0], A[1,1]
        # X_l is a dict of (block_i, block_j) -> coeff
        # Y_l is a dict of (block_i, block_j) -> coeff
        # Strassen formulas for M222
        # w[s][k_prime, i_prime]
        # s=0..6
        # M1: (a+d)(e+h)
        # M2: (c+d)e
        # M3: a(f-h)
        # M4: d(g-e)
        # M5: (a+b)h
        # M6: (c-a)(e+f)
        # M7: (b-d)(g+h)
        
        # Local Strassen W factors (indices k', i')
        # C_ik = sum_j A_ij B_jk
        # Tensor sum A_ij B_jk C_ki
        # W[k', i']
        
        # s=0 (M1):
        # C00 += M1, C11 += M1
        # w[0][0,0]=1, w[0][1,1]=1
        # s=1 (M2):
        # C10 += M2, C11 -= M2
        # w[1][0,1]=1, w[1][1,1]=-1
        # s=2 (M3):
        # C01 += M3, C11 += M3
        # w[2][1,0]=1, w[2][1,1]=1
        # s=3 (M4):
        # C00 += M4, C10 += M4
        # w[3][0,0]=1, w[3][0,1]=1
        # s=4 (M5):
        # C00 -= M5, C01 += M5
        # w[4][0,0]=-1, w[4][1,0]=1
        # s=5 (M6):
        # C11 += M6
        # w[5][1,1]=1
        # s=6 (M7):
        # C00 += M7
        # w[6][0,0]=1
        
        strassen_w = [
            [[1, 0], [0, 1]], # s=0: k'=0,i'=0; k'=1,i'=1
            [[0, 1], [-1, 0]], # s=1: k'=0,i'=1; k'=1,i'=1 (-1)
            [[0, 0], [1, 1]], # s=2: k'=1,i'=0; k'=1,i'=1
            [[1, 0], [0, 1]], # s=3: k'=0,i'=0; k'=0,i'=1
            [[-1, 0], [1, 0]], # s=4: k'=0,i'=0 (-1); k'=1,i'=0
            [[0, 0], [0, 1]], # s=5: k'=1,i'=1
            [[1, 0], [0, 0]]  # s=6: k'=0,i'=0
        ]
        
        # X_l, Y_l
        # l=0: X=A00+A11, Y=B00+B11
        # l=1: X=A10+A11, Y=B00
        # l=2: X=A00, Y=B01-B11
        # l=3: X=A11, Y=B10-B00
        # l=4: X=A00+A01, Y=B11
        # l=5: X=A10-A00, Y=B00+B01
        # l=6: X=A01-A11, Y=B10+B11
        
        # Contributions to blocks (I, K)
        # l=0: (0,0)+1, (1,1)+1
        # l=1: (1,0)+1, (1,1)-1
        # l=2: (0,1)+1, (1,1)+1
        # l=3: (0,0)+1, (1,0)+1
        # l=4: (0,0)-1, (0,1)+1
        # l=5: (1,1)+1
        # l=6: (0,0)+1
        
        contribs = [
            [((0,0), 1), ((1,1), 1)],
            [((1,0), 1), ((1,1), -1)],
            [((0,1), 1), ((1,1), 1)],
            [((0,0), 1), ((1,0), 1)],
            [((0,0), -1), ((0,1), 1)],
            [((1,1), 1)],
            [((0,0), 1)]
        ]
        
        xs = [
            {((0,0), (0,0)): 1, ((1,1), (1,1)): 1}, # l=0: A00, A11
            {((1,0), (1,0)): 1, ((1,1), (1,1)): 1}, # l=1: A10, A11
            {((0,0), (0,0)): 1}, # l=2: A00
            {((1,1), (1,1)): 1}, # l=3: A11
            {((0,0), (0,0)): 1, ((0,1), (0,1)): 1}, # l=4: A00, A01
            {((1,0), (1,0)): 1, ((0,0), (0,0)): -1}, # l=5: A10, -A00
            {((0,1), (0,1)): 1, ((1,1), (1,1)): -1}  # l=6: A01, -A11
        ]
        
        ys = [
            {((0,0), (0,0)): 1, ((1,1), (1,1)): 1}, # l=0: B00, B11
            {((0,0), (0,0)): 1}, # l=1: B00
            {((0,1), (0,1)): 1, ((1,1), (1,1)): -1}, # l=2: B01, -B11
            {((1,0), (1,0)): 1, ((0,0), (0,0)): -1}, # l=3: B10, -B00
            {((1,1), (1,1)): 1}, # l=4: B11
            {((0,0), (0,0)): 1, ((0,1), (0,1)): 1}, # l=5: B00, B01
            {((1,0), (1,0)): 1, ((1,1), (1,1)): 1}  # l=6: B10, B11
        ]
        
        for l in range(7):
            for s in range(7):
                c = col_offset + l * 7 + s
                # Fill U
                for (bi, bj), coeff in xs[l].items():
                    for i in range(2):
                        for j in range(2):
                            gi = bi * 2 + i
                            gj = bj * 2 + j
                            U[gi * m + gj][c] = coeff
                            
                # Fill V
                for (bi, bj), coeff in ys[l].items():
                    for i in range(2):
                        for j in range(2):
                            gi = bi * 2 + i
                            gj = bj * 2 + j
                            gk = offset_k + gj
                            V[gi * p + gk][c] = coeff
                            
                # Fill W
                for (I, K), bcoeff in contribs[l]:
                    w_local = strassen_w[s]
                    for kp in range(2):
                        for ip in range(2):
                            val = w_local[kp][ip] * bcoeff
                            if val != 0:
                                gi = I * 2 + ip
                                gk = K * 2 + kp
                                W[gk * n + gi][c] = val

    # Add first M444 (k=0..3)
    add_strassen_m444(0, 0)
    
    # Add second M444 (k=4..7) for the first 4 cols of B2
    add_strassen_m444(4, 49)
    
    # Add M441 (k=8) for the last col of B2
    # Rank 16
    for i in range(4):
        for j in range(4):
            c = 49 + 49 + i * 4 + j
            U[i * m + j][c] = 1
            V[j * p + 8][c] = 1
            W[8 * n + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
