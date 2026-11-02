"""Program for 3x5 by 5x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}

def _decompose(n, m, p):
    # Base case: trivial decomposition
    if n == 1 or m == 1 or p == 1:
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
    
    # Special case: 2x2x2 Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
        
    # Recursive step: split the largest dimension
    # Heuristic: split p if p >= m and p >= n, else m if m >= n, else n
    # To ensure termination and reasonable splits, we split dimensions > 1.
    
    # We will split p into p1 + p2
    # Or m into m1 + m2
    # Or n into n1 + n2
    
    # Let's split p if p is the largest, otherwise m, otherwise n.
    # For 3x5x5, p=5, m=5, n=3. Split p into 2+3.
    # For 3x3x5, split p into 2+3.
    # For 3x2x5, split p into 2+3.
    # For 2x2x3, split p into 1+2.
    # For 2x3x2, split p into 1+1? No, 2 is even. Split into 1+1.
    
    # General split logic:
    # We want to reduce rank.
    # Rank(n,m,p) <= Rank(n, m1, p) + Rank(n, m2, p) if m = m1 + m2.
    # Rank(n,m,p) <= Rank(n, m, p1) + Rank(n, m, p2) if p = p1 + p2.
    # Rank(n,m,p) <= Rank(n1, m, p) + Rank(n2, m, p) if n = n1 + n2.
    
    # We choose the split that minimizes estimated rank.
    # Estimate: trivial rank is n*m*p. Strassen 2x2x2 is 7.
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Try splitting n
    if n > 1:
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = _decompose(n1, m, p)
            U2, V2, W2 = _decompose(n2, m, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            if r1 + r2 < best_rank:
                # Combine
                # U is (n*m, r1+r2)
                # V is (m*p, r1+r2)
                # W is (p*n, r1+r2)
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                # Fill U1 part
                for i in range(n1 * m):
                    for k in range(r1):
                        U[i][k] = U1[i][k]
                # Fill U2 part (offset in rows)
                for i in range(n2 * m):
                    for k in range(r2):
                        U[n1 * m + i][r1 + k] = U2[i][k]
                        
                # V is same rows, concat columns
                for j in range(m * p):
                    for k in range(r1):
                        V[j][k] = V1[j][k]
                    for k in range(r2):
                        V[j][r1 + k] = V2[j][k]
                        
                # W is same cols, concat rows? No, W is (p*n, rank).
                # W1 is (p*n1, r1), W2 is (p*n2, r2).
                # W[i][k] where i is index in p*n.
                # The index i in W corresponds to (k, i) in C = A B.
                # C has size n x p.
                # W is defined as W[k*n + i, r].
                # So for n1 block, rows 0..p*n1-1.
                # For n2 block, rows p*n1..p*n-1.
                for k in range(p):
                    for i1 in range(n1):
                        for r in range(r1):
                            W[k * n + i1][r] = W1[k * n1 + i1][r]
                    for i2 in range(n2):
                        for r in range(r2):
                            W[k * n + n1 + i2][r1 + r] = W2[k * n2 + i2][r]
                            
                best_U, best_V, best_W = U, V, W
                best_rank = r1 + r2
                
    # Try splitting m
    if m > 1:
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = _decompose(n, m1, p)
            U2, V2, W2 = _decompose(n, m2, p)
            r1 = len(U1[0])
            r2 = len(U2[0])
            if r1 + r2 < best_rank:
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                # U: n*m1 and n*m2
                for i in range(n):
                    for j1 in range(m1):
                        for k in range(r1):
                            U[i * m + j1][k] = U1[i * m1 + j1][k]
                    for j2 in range(m2):
                        for k in range(r2):
                            U[i * m + m1 + j2][r1 + k] = U2[i * m2 + j2][k]
                            
                # V: m1*p and m2*p
                for j1 in range(m1):
                    for k in range(p):
                        for r in range(r1):
                            V[j1 * p + k][r] = V1[j1 * p + k][r]
                for j2 in range(m2):
                    for k in range(p):
                        for r in range(r2):
                            V[(m1 + j2) * p + k][r1 + r] = V2[j2 * p + k][r]
                            
                # W: same rows, concat cols
                for k in range(p):
                    for i in range(n):
                        for r in range(r1):
                            W[k * n + i][r] = W1[k * n + i][r]
                        for r in range(r2):
                            W[k * n + i][r1 + r] = W2[k * n + i][r]
                            
                best_U, best_V, best_W = U, V, W
                best_rank = r1 + r2

    # Try splitting p
    if p > 1:
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = _decompose(n, m, p1)
            U2, V2, W2 = _decompose(n, m, p2)
            r1 = len(U1[0])
            r2 = len(U2[0])
            if r1 + r2 < best_rank:
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                # U: same rows, concat cols
                for i in range(n * m):
                    for k in range(r1):
                        U[i][k] = U1[i][k]
                    for k in range(r2):
                        U[i][r1 + k] = U2[i][k]
                        
                # V: m*p1 and m*p2
                for j in range(m):
                    for k1 in range(p1):
                        for r in range(r1):
                            V[j * p + k1][r] = V1[j * p1 + k1][r]
                    for k2 in range(p2):
                        for r in range(r2):
                            V[j * p + p1 + k2][r1 + r] = V2[j * p2 + k2][r]
                            
                # W: p1*n and p2*n
                for k1 in range(p1):
                    for i in range(n):
                        for r in range(r1):
                            W[k1 * n + i][r] = W1[k1 * n + i][r]
                for k2 in range(p2):
                    for i in range(n):
                        for r in range(r2):
                            W[(p1 + k2) * n + i][r1 + r] = W2[k2 * n + i][r]
                            
                best_U, best_V, best_W = U, V, W
                best_rank = r1 + r2

    return best_U, best_V, best_W

def _strassen_2x2x2():
    # Rank 7 decomposition for 2x2x2
    # U, V, W shapes: (4, 7), (4, 7), (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = [
        [1, 0, 0, 0, 0, 0, 0], # 00
        [0, 1, 0, 0, 0, 0, 0], # 10
        [0, 0, 1, 0, 0, 0, 0], # 01
        [0, 0, 0, 1, 0, 0, 0]  # 11
    ]
    # Wait, Strassen uses linear combinations.
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B12 - B22)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We need to map these to U, V, W.
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    # Let's define the 7 terms.
    # Term 1: (A11+A22)(B11+B22)
    # U: A11=1, A22=1. V: B11=1, B22=1. W: C11+=1, C22+=1.
    # Term 2: (A21+A22)B11
    # U: A21=1, A22=1. V: B11=1. W: C21+=1.
    # Term 3: A11(B21-B11)
    # U: A11=1. V: B21=1, B11=-1. W: C12+=1.
    # Term 4: A22(B12-B22)
    # U: A22=1. V: B12=1, B22=-1. W: C11-=1 (so -1), C22-=1 (so -1)? 
    # Wait, C11 = M1 - M4. So W for M4 on C11 is -1.
    # C22 = M1 - M3 + M5 - M6. M4 is not in C22.
    # Let's re-verify Strassen formulas.
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 - M6
    
    # M1: U(1,1,0,0) V(1,0,0,1) W(1,0,0,1)
    # M2: U(0,1,1,0) V(1,0,0,0) W(0,0,1,0)
    # M3: U(1,0,0,0) V(0,1,-1,0) W(0,1,0,0)
    # M4: U(0,0,0,1) V(0,1,0,-1) W(-1,0,1,0) -> C11 gets -1, C21 gets 1.
    # M5: U(1,0,1,0) V(0,0,0,1) W(0,1,0,1) -> C12 gets 1, C22 gets 1.
    # M6: U(0,1,-1,0) V(1,1,0,0) W(1,0,0,-1) -> C11 gets 1, C22 gets -1.
    # M7: U(0,0,1,-1) V(0,0,1,1) W(-1,0,0,0) -> C11 gets -1.
    
    # Let's build the matrices.
    # U is 4x7. Rows: A11, A12, A21, A22. (Index i*2+j)
    # 0: A11, 1: A12, 2: A21, 3: A22.
    # V is 4x7. Rows: B11, B12, B21, B22. (Index j*2+k)
    # 0: B11, 1: B12, 2: B21, 3: B22.
    # W is 4x7. Rows: C11, C12, C21, C22. (Index k*2+i)
    # 0: C11 (k=0,i=0), 1: C12 (k=0,i=1), 2: C21 (k=1,i=0), 3: C22 (k=1,i=1).
    
    U = [
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, -1], # A21
        [1, 1, 0, 1, 0, 0, 0]   # A22
    ]
    V = [
        [1, 1, -1, 0, 0, 1, 0], # B11
        [0, 0, 0, 1, 0, 1, 0],  # B12
        [0, 0, 1, 0, 0, 0, 1],  # B21
        [1, 0, 0, -1, 1, 0, 1]  # B22
    ]
    W = [
        [1, 0, 0, -1, 0, 1, -1], # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, 0, 0, 0, 1, -1, 0]   # C22
    ]
    return U, V, W
# EVOLVE-BLOCK-END
