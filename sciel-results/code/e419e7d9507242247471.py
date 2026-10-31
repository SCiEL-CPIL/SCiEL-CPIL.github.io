import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get block indices
    # Block (bp, bq) corresponds to rows 2*bp to 2*bp+1, cols 2*bq to 2*bq+1
    
    # Define the 7 outer X and Y structures
    # Each is a list of 2x2 matrices of linear forms (lists of 16 coeffs)
    # We will generate them on the fly
    
    # Inner Strassen coefficients for Z[u,v] from S[t]
    # Z00 = S1 + S4 - S5 + S7
    # Z01 = S3 + S5
    # Z10 = S2 + S4
    # Z11 = S1 - S2 + S3 + S6
    C_inner = [
        [1, 0, 0, 1, -1, 0, 1], # u=0, v=0
        [0, 0, 1, 0, 1, 0, 0], # u=0, v=1
        [0, 1, 0, 1, 0, 0, 0], # u=1, v=0
        [1, -1, 1, 0, 0, 1, 0]  # u=1, v=1
    ]
    
    # Outer Strassen coefficients for C[p,q] from M[k]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    D_outer = [
        [1, 0, 0, 1, -1, 0, 1], # p=0, q=0
        [0, 0, 1, 0, 1, 0, 0], # p=0, q=1
        [0, 1, 0, 1, 0, 0, 0], # p=1, q=0
        [1, -1, 1, 0, 0, 1, 0]  # p=1, q=1
    ]
    
    # Define the linear forms for X_k and Y_k
    # We need a function to create a linear form (list of 16 zeros) and add a block to it
    def make_block_form(bp, bq, coeff=1):
        form = [0] * 16
        for u in range(2):
            for v in range(2):
                i = 2 * bp + u
                j = 2 * bq + v
                form[i * 4 + j] += coeff
        return form

    # Define the 7 X and Y as lists of 4 linear forms (for 2x2 matrix entries)
    # X_k[u, v] is a linear form
    X_structs = []
    Y_structs = []
    
    # k=0: X = A00 + A11, Y = B00 + B11
    X0 = [
        make_block_form(0,0,1) if False else None # Placeholder
    ]
    # Let's just build them explicitly in a loop or list
    
    def build_struct(defs):
        # defs is a list of 4 tuples (bp, bq, coeff) for u,v in 0,1,0,1
        # Actually, the structure is the same for all u,v if it's a block sum?
        # No, X is a 2x2 matrix.
        # For X = A00 + A11, X[0,0] = A00[0,0] + A11[0,0], etc.
        # So each entry X[u,v] is the sum of the corresponding entries of the blocks.
        
        # Let's define the blocks involved for each k
        # k=0: X_blocks = [(0,0,1), (1,1,1)], Y_blocks = [(0,0,1), (1,1,1)]
        # k=1: X_blocks = [(1,0,1), (1,1,1)], Y_blocks = [(0,0,1)]
        # k=2: X_blocks = [(0,0,1)], Y_blocks = [(1,0,1), (1,1,-1)]
        # k=3: X_blocks = [(1,1,1)], Y_blocks = [(0,1,1), (0,0,-1)]
        # k=4: X_blocks = [(0,0,1), (0,1,1)], Y_blocks = [(1,1,1)]
        # k=5: X_blocks = [(1,0,1), (0,0,-1)], Y_blocks = [(0,0,1), (0,1,1)]
        # k=6: X_blocks = [(0,1,1), (1,1,-1)], Y_blocks = [(1,0,1), (1,1,1)]
        
        pass

    X_defs = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    Y_defs = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)]
    ]

    # Build X_k and Y_k matrices of linear forms
    X_mats = []
    Y_mats = []
    for k in range(7):
        X_mat = [[None for _ in range(2)] for _ in range(2)]
        Y_mat = [[None for _ in range(2)] for _ in range(2)]
        
        for u in range(2):
            for v in range(2):
                xf = [0]*16
                for bp, bq, c in X_defs[k]:
                    xf[ (2*bp+u)*4 + (2*bq+v) ] += c
                X_mat[u][v] = xf
                
                yf = [0]*16
                for bp, bq, c in Y_defs[k]:
                    yf[ (2*bp+u)*4 + (2*bq+v) ] += c
                Y_mat[u][v] = yf
                
        X_mats.append(X_mat)
        Y_mats.append(Y_mat)

    # Inner Strassen linear combinations for S_t
    # S1 = X00 + X11, Y00 + Y11
    # S2 = X10 + X11, Y00
    # S3 = X00, Y10 - Y11
    # S4 = X11, Y01 - Y00
    # S5 = X00 + X01, Y11
    # S6 = X10 - X00, Y00 + Y01
    # S7 = X01 - X11, Y10 + Y11
    
    def add_forms(f1, f2, c1=1, c2=1):
        return [c1*a + c2*b for a, b in zip(f1, f2)]
        
    def sub_forms(f1, f2):
        return [a - b for a, b in zip(f1, f2)]

    # We will iterate k and t
    for k in range(7):
        X_mat = X_mats[k]
        Y_mat = Y_mats[k]
        
        # Define the 7 S_t linear forms
        S_X = []
        S_Y = []
        
        # t=0
        S_X.append(add_forms(X_mat[0][0], X_mat[1][1]))
        S_Y.append(add_forms(Y_mat[0][0], Y_mat[1][1]))
        # t=1
        S_X.append(add_forms(X_mat[1][0], X_mat[1][1]))
        S_Y.append(Y_mat[0][0][:]) # copy
        # t=2
        S_X.append(X_mat[0][0][:])
        S_Y.append(sub_forms(Y_mat[1][0], Y_mat[1][1]))
        # t=3
        S_X.append(X_mat[1][1][:])
        S_Y.append(sub_forms(Y_mat[0][1], Y_mat[0][0]))
        # t=4
        S_X.append(add_forms(X_mat[0][0], X_mat[0][1]))
        S_Y.append(Y_mat[1][1][:])
        # t=5
        S_X.append(sub_forms(X_mat[1][0], X_mat[0][0]))
        S_Y.append(add_forms(Y_mat[0][0], Y_mat[0][1]))
        # t=6
        S_X.append(sub_forms(X_mat[0][1], X_mat[1][1]))
        S_Y.append(add_forms(Y_mat[1][0], Y_mat[1][1]))
        
        for t in range(7):
            col = k * 7 + t
            LX = S_X[t]
            LY = S_Y[t]
            
            # Fill U
            for idx in range(16):
                U[idx][col] = LX[idx]
                
            # Fill V
            for idx in range(16):
                V[idx][col] = LY[idx]
                
            # Fill W
            for I in range(4):
                for K in range(4):
                    p = I // 2
                    u = I % 2
                    q = K // 2
                    v = K % 2
                    
                    # Coefficient from inner Strassen
                    c_inner_val = C_inner[u*2+v][t] # C_inner indexed by u,v then t
                    # Wait, C_inner is defined as list of 4 lists, each of length 7.
                    # Index 0: u=0,v=0. Index 1: u=0,v=1. Index 2: u=1,v=0. Index 3: u=1,v=1.
                    # So u*2+v works.
                    
                    # Coefficient from outer Strassen
                    c_outer_val = D_outer[p*2+q][k] # D_outer indexed by p,q then k
                    
                    w_val = c_inner_val * c_outer_val
                    W[K*4 + I][col] = w_val

    return {"U": U, "V": V, "W": W}
