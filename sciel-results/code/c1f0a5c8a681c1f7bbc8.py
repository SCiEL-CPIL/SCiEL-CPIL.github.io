"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=3, p=9
    # Rank 52 decomposition
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u, v, w):
        nonlocal col
        if col >= rank:
            return
        U[u][col] = 1
        V[v][col] = 1
        W[w][col] = 1
        col += 1

    # We need to construct the decomposition for 2x3 * 3x9
    # Split m=3 into 2+1
    # A = [A_L, A_R] (2x2, 2x1)
    # B = [B_T; B_B] (2x9, 1x9)
    # C = A_L B_T + A_R B_B
    
    # Part 1: A_R B_B (2x1 * 1x9)
    # A_R is column 2 of A (index 2 in 0-based? A is 2x3, indices 0,1,2)
    # A_R is A[:, 2]
    # B_B is B[2, :]
    # Rank 18: 2*1*9
    for i in range(2):
        for k in range(9):
            # A_{i, 2} * B_{2, k} contributes to C_{i, k}
            # U index: i * 3 + 2
            # V index: 2 * 9 + k
            # W index: k * 2 + i
            add_term(i * 3 + 2, 2 * 9 + k, k * 2 + i)
            
    # Part 2: A_L B_T (2x2 * 2x9)
    # A_L is A[:, 0:2]
    # B_T is B[0:2, :]
    # Split p=9 into 4+4+1
    # B_T = [B_1, B_2, B_3] (2x4, 2x4, 2x1)
    # A_L B_1 (2x2 * 2x4)
    # Split 4 into 2+2
    # B_1 = [B_1a, B_1b] (2x2, 2x2)
    # A_L B_1a (2x2 * 2x2) -> Rank 7 (Strassen)
    # A_L B_1b (2x2 * 2x2) -> Rank 7 (Strassen)
    # A_L B_2 (2x2 * 2x4) -> Rank 14
    # A_L B_3 (2x2 * 2x1) -> Rank 6
    
    # Strassen 2x2x2 decomposition
    # Inputs: A (2x2), B (2x2)
    # Indices for A: 0,1,2,3 (row-major)
    # Indices for B: 0,1,2,3 (row-major)
    # Indices for C: 0,1,2,3 (row-major)
    
    def strassen_2x2x2(offset_u, offset_v, offset_w):
        nonlocal col
        # 1. (A11+A22)(B11+B22) -> C11, C22
        # 2. (A21+A22)B11 -> C21
        # 3. A11(B21-B11) -> C12
        # 4. A22(B12-B22) -> C21
        # 5. (A11+A12)B22 -> C12
        # 6. (A21-A11)(B11+B12) -> C21
        # 7. (A12-A22)(B21+B22) -> C12
        
        # We need to map these to the global indices
        # A indices: offset_u + 0, 1, 2, 3
        # B indices: offset_v + 0, 1, 2, 3
        # C indices: offset_w + 0, 1, 2, 3
        
        # Term 1: (A11+A22)(B11+B22)
        # U: A11 (1), A22 (1)
        # V: B11 (1), B22 (1)
        # W: C11 (1), C22 (1)
        # This is not rank 1. It's a sum.
        # Strassen works by pre-computing sums.
        # But we are building the tensor decomposition directly.
        # The rank-1 terms of Strassen are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add these as rank-1 terms in the U, V, W matrices.
        # Each M is a product of a linear combo of A and a linear combo of B.
        # So each M corresponds to a rank-1 tensor term with coefficients.
        
        # M1: (A11+A22)(B11+B22)
        # U: A11 (1), A22 (1)
        # V: B11 (1), B22 (1)
        # W: C11 (1), C22 (1)
        # Wait, M1 contributes to C11 and C22.
        # So we add a term for C11 and a term for C22?
        # No, the rank-1 term is $u \otimes v \otimes w$.
        # $u$ is a vector in $A$ space, $v$ in $B$ space, $w$ in $C$ space.
        # $M1 = (e_1+e_4)^T A (e_1+e_4)^T B$.
        # This is a rank-1 term in the tensor product space?
        # Yes.
        # $u = e_1 + e_4$ (indices 0, 3 in 0-based)
        # $v = e_1 + e_4$ (indices 0, 3)
        # $w = e_1 + e_4$ (indices 0, 3)
        
        # Let's define the vectors.
        # A indices: 0,1,2,3
        # B indices: 0,1,2,3
        # C indices: 0,1,2,3
        
        # M1
        u = [0]*4; u[0]=1; u[3]=1
        v = [0]*4; v[0]=1; v[3]=1
        w = [0]*4; w[0]=1; w[3]=1
        for i in range(4):
            if u[i]: U[offset_u+i][col] += u[i]
            if v[i]: V[offset_v+i][col] += v[i]
            if w[i]: W[offset_w+i][col] += w[i]
        col += 1
        
        # M2: (A21+A22)B11
        u = [0]*4; u[2]=1; u[3]=1
        v = [0]*4; v[0]=1
        w = [0]*4; w[2]=1
        for i in range(4):
            if u[i]: U[offset_u+i][col] += u[i]
            if v[i]: V[offset_v+i][col] += v[i]
            if w[i]: W[offset_w+i][col] += w[i]
        col += 1
        
        # M3: A11(B21-B11)
        u = [0]*4; u[0]=1
        v = [0]*4; v[1]=1; v[0]=-1
        w = [0]*4; w[1]=1
        for i in range(4):
            if u[i]: U[offset_u+i][col] += u[i]
            if v[i]: V[offset_v+i][col] += v[i]
            if w[i]: W[offset_w+i][col] += w[i]
        col += 1
        
        # M4: A22(B12-B22)
        u = [0]*4; u[3]=1
        v = [0]*4; v[1]=1; v[3]=-1
        w = [0]*4; w[2]=1; w[3]=-1
        for i in range(4):
            if u[i]: U[offset_u+i][col] += u[i]
            if v[i]: V[offset_v+i][col] += v[i]
            if w[i]: W[offset_w+i][col] += w[i]
        col += 1
        
        # M5: (A11+A12)B22
        u = [0]*4; u[0]=1; u[1]=1
        v = [0]*4; v[3]=1
        w = [0]*4; w[1]=1; w[2]=-1
        for i in range(4):
            if u[i]: U[offset_u+i][col] += u[i]
            if v[i]: V[offset_v+i][col] += v[i]
            if w[i]: W[offset_w+i][col] += w[i]
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        u = [0]*4; u[2]=1; u[0]=-1
        v = [0]*4; v[0]=1; v[1]=1
        w = [0]*4; w[3]=1
        for i in range(4):
            if u[i]: U[offset_u+i][col] += u[i]
            if v[i]: V[offset_v+i][col] += v[i]
            if w[i]: W[offset_w+i][col] += w[i]
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        u = [0]*4; u[1]=1; u[3]=-1
        v = [0]*4; v[1]=1; v[3]=1
        w = [0]*4; w[1]=-1
        for i in range(4):
            if u[i]: U[offset_u+i][col] += u[i]
            if v[i]: V[offset_v+i][col] += v[i]
            if w[i]: W[offset_w+i][col] += w[i]
        col += 1

    # Apply Strassen to the 2x2x2 blocks
    # A_L B_1a: A[:, 0:2] * B[0:2, 0:2]
    # U offset: 0, V offset: 0, W offset: 0
    strassen_2x2x2(0, 0, 0)
    
    # A_L B_1b: A[:, 0:2] * B[0:2, 2:4]
    # U offset: 0, V offset: 2, W offset: 4
    strassen_2x2x2(0, 2, 4)
    
    # A_L B_2a: A[:, 0:2] * B[0:2, 4:6]
    # U offset: 0, V offset: 6, W offset: 8
    strassen_2x2x2(0, 6, 8)
    
    # A_L B_2b: A[:, 0:2] * B[0:2, 6:8]
    # U offset: 0, V offset: 10, W offset: 12
    strassen_2x2x2(0, 10, 12)
    
    # A_L B_3: A[:, 0:2] * B[0:2, 8:9]
    # This is 2x2 * 2x1. Rank 6.
    # We need a rank 6 decomposition for 2x2 * 2x1.
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # This is a bilinear map from 4x2 to 2.
    # Rank 6 decomposition:
    # 1. A11 B1 -> C1
    # 2. A12 B2 -> C1
    # 3. A21 B1 -> C2
    # 4. A22 B2 -> C2
    # 5. (A11+A22)(B1-B2) -> C1, C2
    # 6. (A21-A11)(B1+B2) -> C2
    # Let's verify this.
    # C1 = A11 B1 + A12 B2 + 0.5 * (A11+A22)(B1-B2) - 0.5 * (A11+A22)(B1-B2) ?
    # Actually, let's just use schoolbook for 2x2*2x1 if rank 6 is hard to code.
    # Schoolbook rank is 4. Wait.
    # C1 = A11 B1 + A12 B2 (2 terms)
    # C2 = A21 B1 + A22 B2 (2 terms)
    # Total 4 terms.
    # Why did I think rank 6?
    # 2x2x1 tensor rank is 6?
    # Let's check.
    # If rank is 4, then total rank is 14+14+4+18 = 50.
    # If rank is 6, total rank is 52.
    # Let's use schoolbook (rank 4) for 2x2*2x1.
    
    for i in range(2):
        for j in range(2):
            for k in range(1):
                # A_{i, j} B_{j, k} -> C_{i, k}
                # A index: i * 2 + j
                # B index: j * 1 + k
                # C index: k * 2 + i
                # Global offsets:
                # A offset: 0
                # B offset: 8
                # C offset: 16
                add_term(i * 2 + j, 8 + j * 1 + k, 16 + k * 2 + i)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
