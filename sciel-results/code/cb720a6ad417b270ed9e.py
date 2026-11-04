import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 4x4 matrix multiplication.
    # Rank is 49 (7 * 7).
    n, m, p = DIMENSIONS
    
    # Define the 2x2 Strassen decomposition components
    # Indices for 2x2:
    # U (A input): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V (B input): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W (C output): k*2+i. 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2: (A10+A11)B00 -> C10+1, C11-1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # M3: A00(B10-B11) -> C01+1, C11+1
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # M4: A11(B01-B00) -> C00+1, C10+1
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # M5: (A00+A01)B11 -> C01+1, C00-1
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[2][4] = 1; W2[0][4] = -1
    
    # M6: (A10-A00)(B00+B01) -> C11+1
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00+1
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Kronecker product logic
    # i = 2*i1 + i2
    # c = 7*c1 + c2
    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for c1 in range(7):
                u1_val = U2[i1][c1]
                if u1_val == 0: continue
                for c2 in range(7):
                    u2_val = U2[i2][c2]
                    if u2_val == 0: continue
                    c = 7 * c1 + c2
                    U[i][c] = u1_val * u2_val

    for j1 in range(2):
        for j2 in range(2):
            j = 2 * j1 + j2
            for c1 in range(7):
                v1_val = V2[j1][c1]
                if v1_val == 0: continue
                for c2 in range(7):
                    v2_val = V2[j2][c2]
                    if v2_val == 0: continue
                    c = 7 * c1 + c2
                    V[j][c] = v1_val * v2_val

    for k1 in range(2):
        for i1 in range(2):
            k = 2 * k1 + k2 # Wait, W index is k*n+i.
            # W rows are indexed by (k, i).
            # k = 2*k1 + k2? No, k is the row index of C (from B), i is col index of C (from A).
            # In 4x4, k in 0..3, i in 0..3.
            # k = 2*k1 + k2, i = 2*i1 + i2.
            # W index = k*4 + i.
            # W2 index = k2*2 + i2.
            # c1 corresponds to the "outer" Strassen step (block level), c2 to "inner" (scalar level).
            # The decomposition is tensor product.
            # W[k, c] = W2[k1, c1] * W2[k2, c2]?
            # Let's check indices.
            # Outer level: k1, i1. Inner level: k2, i2.
            # Global k = 2*k1 + k2.
            # Global i = 2*i1 + i2.
            # Global W index = (2*k1 + k2)*4 + (2*i1 + i2) = 8*k1 + 4*k2 + 4*i1 + i2.
            # W2 index for outer = k1*2 + i1.
            # W2 index for inner = k2*2 + i2.
            
            for i2 in range(2):
                k = 2 * k1 + k2 # This is wrong in the loop structure above.
                
    # Correct loop for W:
    for k1 in range(2):
        for i1 in range(2):
            w1_idx = k1 * 2 + i1
            for c1 in range(7):
                w1_val = W2[w1_idx][c1]
                if w1_val == 0: continue
                for k2 in range(2):
                    for i2 in range(2):
                        w2_idx = k2 * 2 + i2
                        w2_val = W2[w2_idx][c1] # Wait, c2 is the inner column index.
                        # The column index c is 7*c1 + c2.
                        # So we need to iterate c2 as well.
                        pass
                        
    # Let's rewrite the W construction properly.
    for c1 in range(7):
        for c2 in range(7):
            c = 7 * c1 + c2
            for k1 in range(2):
                for i1 in range(2):
                    w1_idx = k1 * 2 + i1
                    w1_val = W2[w1_idx][c1]
                    if w1_val == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2_idx = k2 * 2 + i2
                            w2_val = W2[w2_idx][c2]
                            if w2_val == 0: continue
                            k = 2 * k1 + k2
                            i = 2 * i1 + i2
                            w_idx = k * n + i
                            W[w_idx][c] = w1_val * w2_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
