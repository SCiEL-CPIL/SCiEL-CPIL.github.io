import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    # We decompose M_{2,3,6} as M_{2,3,2} \otimes M_{1,1,3}
    # Rank(M_{1,1,3}) = 3 (Schoolbook)
    # Rank(M_{2,3,2}) = 11 (Strassen for 2x2x2 part + Schoolbook for 2x1x2 part)
    
    # Sub-problem 2: M_{1,1,3}
    # n2=1, m2=1, p2=3
    # U2, V2, W2 shapes: (1, 3), (3, 3), (3, 1) ? No.
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # For M_{1,1,3}:
    # n=1, m=1, p=3.
    # U: (1*1, 3) = (1, 3)
    # V: (1*3, 3) = (3, 3)
    # W: (3*1, 3) = (3, 3)
    # Rank 3.
    
    U2 = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]] # Wait, U is (n*m, rank). n*m=1.
    # Correct shapes for M_{n,m,p}:
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # M_{1,1,3}:
    # U2: (1, 3)
    # V2: (3, 3)
    # W2: (3, 1)
    
    U2 = [[1, 1, 1]] # A_0
    V2 = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]] # B_0, B_1, B_2
    W2 = [[1], [1], [1]] # C_0, C_1, C_2? 
    # C_k = A_0 B_k.
    # Term k: U2[0, k] * V2[k, k] * W2[k, 0] = 1 * 1 * 1 = 1.
    # Sum over k? No, Kronecker product structure.
    
    # Let's build the full Kronecker product.
    # T = T1 \otimes T2.
    # U = U1 \otimes U2 ?
    # Index mapping:
    # i = i1 * n2 + i2
    # j = j1 * m2 + j2
    # k = k1 * p2 + k2
    
    # T1: M_{2,3,2}. n1=2, m1=3, p1=2.
    # T2: M_{1,1,3}. n2=1, m2=1, p2=3.
    
    # Rank R1 = 11, R2 = 3. Total Rank R = 33.
    
    # Construct T1 (M_{2,3,2})
    # Split m=3 into m1=2, m2=1.
    # T1 = T1a (M_{2,2,2}) \oplus T1b (M_{2,1,2}).
    # T1a Rank 7 (Strassen).
    # T1b Rank 4 (Schoolbook).
    
    # Strassen for M_{2,2,2}:
    # U1a, V1a, W1a.
    # U1a: (4, 7)
    # V1a: (4, 7)
    # W1a: (4, 7)
    
    # Schoolbook for M_{2,1,2}:
    # U1b: (2, 4)
    # V1b: (2, 4)
    # W1b: (4, 4)
    
    # Combine T1a and T1b to form T1.
    # T1 U: (n1*m1, R1) = (6, 11)
    # T1 V: (m1*p1, R1) = (6, 11)
    # T1 W: (p1*n1, R1) = (4, 11)
    
    # This is getting complex to hardcode.
    # I will generate the matrices programmatically.
    
    def strassen_2x2x2():
        # Returns U, V, W for M_{2,2,2}
        # U: (4, 7), V: (4, 7), W: (4, 7)
        U = [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0]
        ]
        V = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0],
            [1, -1, 0, 0, 1, 1, 0]
        ]
        W = [
            [0, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, -1],
            [0, 1, 1, 0, 0, 0, 0]
        ]
        return U, V, W

    def schoolbook_nm_p(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Build T1: M_{2,3,2}
    # n1=2, m1=3, p1=2
    # Split m1 into 2 and 1.
    # T1a: M_{2,2,2} (indices i in 0..1, j in 0..1, k in 0..1)
    # T1b: M_{2,1,2} (indices i in 0..1, j in 2, k in 0..1)
    
    U1a, V1a, W1a = strassen_2x2x2()
    U1b, V1b, W1b = schoolbook_nm_p(2, 1, 2)
    
    # Combine U1a and U1b
    # U1a shape: (4, 7). Indices i*m1 + j where j in 0..1.
    # U1b shape: (2, 4). Indices i*m1 + j where j in 2.
    # Total U1 shape: (6, 11).
    
    R1a = 7
    R1b = 4
    R1 = R1a + R1b
    
    U1 = [[0] * R1 for _ in range(6)]
    V1 = [[0] * R1 for _ in range(6)]
    W1 = [[0] * R1 for _ in range(4)]
    
    # Fill T1a part
    for i in range(4):
        for r in range(R1a):
            U1[i][r] = U1a[i][r]
            V1[i][r] = V1a[i][r]
            W1[i][r] = W1a[i][r]
            
    # Fill T1b part
    # U1b indices: i in 0..1, j in 2. Row index i*3 + 2.
    # V1b indices: j in 2, k in 0..1. Row index 2*2 + k = 4+k.
    # W1b indices: k in 0..1, i in 0..1. Row index k*2 + i.
    
    for i in range(2):
        for k in range(2):
            for r in range(R1b):
                u_idx = i * 3 + 2
                v_idx = 2 * 2 + k
                w_idx = k * 2 + i
                col_offset = R1a + r
                U1[u_idx][col_offset] = U1b[i][r]
                V1[v_idx][col_offset] = V1b[2 * 1 + k][r] # V1b row is j*p+k. j=0 in sub-problem, so 0*1+k=k.
                # Wait, schoolbook_nm_p(2,1,2) V shape is (1*2, 4) = (2,4).
                # V1b[j*2+k][r]. j is 0. So V1b[k][r].
                V1[v_idx][col_offset] = V1b[k][r]
                W1[w_idx][col_offset] = W1b[k * 2 + i][r]
                
    # Build T2: M_{1,1,3}
    # n2=1, m2=1, p2=3
    U2, V2, W2 = schoolbook_nm_p(1, 1, 3)
    R2 = 3
    
    # Kronecker Product
    # U = U1 \otimes U2
    # V = V1 \otimes V2
    # W = W1 \otimes W2
    
    # Index mapping for Kronecker:
    # U row: i1*m1*n2 + i2*m2 ?
    # Standard Kronecker index: i = i1 * n2 + i2.
    # U1 row index: i1 * m1 + j1.
    # U2 row index: i2 * m2 + j2.
    # Combined row index: (i1 * m1 + j1) * (n2 * m2) + (i2 * m2 + j2) ?
    # No.
    # Tensor product of matrices A (size I x J) and B (size K x L) is size (I*K) x (J*L).
    # Row index r = r_A * K + r_B.
    # Col index c = c_A * L + c_B.
    
    # U1 size: (6, 11). U2 size: (1, 3).
    # U size: (6*1, 11*3) = (6, 33).
    
    U = [[0] * 33 for _ in range(6)]
    V = [[0] * 33 for _ in range(18)] # V1 (6,11) x V2 (3,3) -> (18, 33)
    W = [[0] * 33 for _ in range(12)] # W1 (4,11) x W2 (3,3) -> (12, 33)
    
    # Fill U
    for r1 in range(6):
        for c1 in range(11):
            val1 = U1[r1][c1]
            if val1 != 0:
                for r2 in range(1):
                    for c2 in range(3):
                        val2 = U2[r2][c2]
                        if val2 != 0:
                            r = r1 * 1 + r2
                            c = c1 * 3 + c2
                            U[r][c] += val1 * val2
                            
    # Fill V
    for r1 in range(6):
        for c1 in range(11):
            val1 = V1[r1][c1]
            if val1 != 0:
                for r2 in range(3):
                    for c2 in range(3):
                        val2 = V2[r2][c2]
                        if val2 != 0:
                            r = r1 * 3 + r2
                            c = c1 * 3 + c2
                            V[r][c] += val1 * val2
                            
    # Fill W
    for r1 in range(4):
        for c1 in range(11):
            val1 = W1[r1][c1]
            if val1 != 0:
                for r2 in range(3):
                    for c2 in range(3):
                        val2 = W2[r2][c2]
                        if val2 != 0:
                            r = r1 * 3 + r2
                            c = c1 * 3 + c2
                            W[r][c] += val1 * val2
                            
    return {"U": U, "V": V, "W": W}
