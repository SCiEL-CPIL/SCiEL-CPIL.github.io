import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = 2, 3, 5
    # Rank 28 decomposition
    # Part 1: A[:,:2] * B[:2,:] (2x2 x 2x5) using Strassen on 2x2x2x2 blocks
    # Part 2: A[:,2] * B[2,:] (2x1 x 1x5) schoolbook
    
    U = []
    V = []
    W = []
    
    # Part 2: 2x1 x 1x5, rank 10
    for i in range(2):
        for k in range(5):
            u = [0]*6
            u[i*3+2] = 1  # A[i,2]
            v = [0]*15
            v[2*5+k] = 1  # B[2,k]
            w = [0]*10
            w[k*2+i] = 1  # C[i,k]
            U.append(u)
            V.append(v)
            W.append(w)
    
    # Part 1: 2x2 x 2x5
    # B[:2,:] split into B[:2,:2] (2x2), B[:2,2:4] (2x2), B[:2,4:5] (2x1)
    
    # Strassen for 2x2 x 2x2, applied to B cols 0,1
    # A1 = A[:2,:2], B1a = B[:2,:2]
    # Strassen terms:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[1,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[1,0])
    # M7 = (A[0,1]-A[1,1])*(B[0,1]+B[1,1])
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    def add_strassen(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """Add Strassen decomposition for A[A_rows, A_cols] * B[B_rows, B_cols] -> C[C_rows, C_cols]"""
        # A is 2x2: indices (A_rows[0], A_cols[0]), (A_rows[0], A_cols[1]), (A_rows[1], A_cols[0]), (A_rows[1], A_cols[1])
        # B is 2x2: indices (B_rows[0], B_cols[0]), ...
        # C is 2x2: indices (C_rows[0], C_cols[0]), ...
        
        a00, a01 = (A_rows[0], A_cols[0]), (A_rows[0], A_cols[1])
        a10, a11 = (A_rows[1], A_cols[0]), (A_rows[1], A_cols[1])
        b00, b01 = (B_rows[0], B_cols[0]), (B_rows[0], B_cols[1])
        b10, b11 = (B_rows[1], B_cols[0]), (B_rows[1], B_cols[1])
        c00, c01 = (C_rows[0], C_cols[0]), (C_rows[0], C_cols[1])
        c10, c11 = (C_rows[1], C_cols[0]), (C_rows[1], C_cols[1])
        
        def make_u(indices):
            u = [0]*6
            for (r, c) in indices:
                u[r*3+c] = 1
            return u
        
        def make_v(indices):
            v = [0]*15
            for (r, c) in indices:
                v[r*5+c] = 1
            return v
        
        def make_w(indices):
            w = [0]*10
            for (r, c) in indices:
                w[c*2+r] = 1
            return w
        
        # M1
        u = make_u([a00, a11])
        v = make_v([b00, b11])
        w = make_w([c00, c11])
        U.append(u); V.append(v); W.append(w)
        
        # M2
        u = make_u([a10, a11])
        v = make_v([b00])
        w = make_w([c10])
        U.append(u); V.append(v); W.append(w)
        
        # M3
        u = make_u([a00])
        v = make_v([b10, (-1, b11)])
        # Hmm, this doesn't work with the simple make_v. I need to handle coefficients.
        
    # This approach is getting messy. Let me use a different method.
