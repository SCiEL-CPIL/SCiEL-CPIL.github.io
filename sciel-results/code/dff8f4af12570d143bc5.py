import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Rank 55 decomposition:
    # 5 blocks of Strassen (2x2x2) for j in {0,1}, k pairs (0,1)..(8,9) -> 5 * 7 = 35
    # 1 block of Schoolbook (2x1x10) for j=2 -> 20
    # Total 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 20 rows
    
    col = 0
    
    # Helper to add a term to the decomposition
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        if col >= rank:
            return
        for idx in u_indices:
            U[idx][col] += 1
        for idx in v_indices:
            V[idx][col] += 1
        for idx in w_indices:
            W[idx][col] += 1
        col += 1

    # Strassen for 2x2x2
    # A: 2x2, B: 2x2, C: 2x2
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A11-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
    
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # Indices mapping:
        # A[i,j] -> U[i*3+j]
        # B[j,k] -> V[j*10+k]
        # C[i,k] -> W[k*2+i]
        
        # M1
        add_term([0*3+0, 1*3+1], [0*10+k0, 1*10+k1], [k0*2+0, k1*2+1])
        # M2
        add_term([1*3+0, 1*3+1], [0*10+k0], [k0*2+1, k1*2+1])
        # Note: M2 in W has -1 for C11? 
        # Standard Strassen:
        # C11 = M1 - M2 + M3 + M6 - M7
        # C10 = M2 + M4
        # C01 = M3 + M5
        # C00 = M1 + M4 - M5 + M7
        
        # Let's re-verify the signs in the seed code.
        # Seed M2: W[k0*2+1]=1, W[k1*2+1]=-1.
        # C10 (k0, i=1) += 1. C11 (k1, i=1) += -1.
        # This matches C10 += M2, C11 -= M2.
        
        # M3
        # Seed: U[0*3+0], V[0*10+k1], V[1*10+k1](-1).
        # W[k1*2+0]=1, W[k1*2+1]=1.
        # C01 += 1, C11 += 1.
        # Matches C01 += M3, C11 += M3.
        
        # M4
        # Seed: U[1*3+1], V[1*10+k0], V[0*10+k0](-1).
        # W[k0*2+0]=1, W[k0*2+1]=1.
        # C00 += 1, C10 += 1.
        # Matches C00 += M4, C10 += M4.
        
        # M5
        # Seed: U[0*3+0], U[0*3+1], V[1*10+k1].
        # W[k0*2+0]=-1, W[k1*2+0]=1.
        # C00 -= 1, C01 += 1.
        # Matches C00 -= M5, C01 += M5.
        
        # M6
        # Seed: U[1*3+1], U[0*3+0](-1), V[0*10+k0], V[0*10+k1].
        # W[k1*2+1]=1.
        # C11 += 1.
        # Matches C11 += M6.
        
        # M7
        # Seed: U[0*3+1], U[1*3+1](-1), V[1*10+k0], V[1*10+k1].
        # W[k0*2+0]=1, W[k1*2+1]=-1.
        # C00 += 1, C11 -= 1.
        # Matches C00 += M7, C11 -= M7.
        
        # The seed code logic is correct. I will replicate it directly for safety.
        # I will not use the add_term helper for the complex ones to avoid sign errors in my refactor.
        
    # Re-doing the loop with explicit assignment as in seed to ensure correctness
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        # M1
        U[0*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*10+k0][col] = 1
        V[1*10+k1][col] = 1
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M2
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*10+k0][col] = 1
        W[k0*2+1][col] = 1
        W[k1*2+1][col] = -1
        col += 1
        
        # M3
        U[0*3+0][col] = 1
        V[0*10+k1][col] = 1
        V[1*10+k1][col] = -1
        W[k1*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M4
        U[1*3+1][col] = 1
        V[1*10+k0][col] = 1
        V[0*10+k0][col] = -1
        W[k0*2+0][col] = 1
        W[k0*2+1][col] = 1
        col += 1
        
        # M5
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[1*10+k1][col] = 1
        W[k0*2+0][col] = -1
        W[k1*2+0][col] = 1
        col += 1
        
        # M6
        U[1*3+1][col] = 1
        U[0*3+0][col] = -1
        V[0*10+k0][col] = 1
        V[0*10+k1][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M7
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        V[1*10+k0][col] = 1
        V[1*10+k1][col] = 1
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = -1
        col += 1
        
    # Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            U[i*3+2][col] = 1
            V[2*10+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Simple verification
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Use a single deterministic test case to avoid random overhead and flakiness
        A = [[1, 2, 3], [4, 5, 6]]
        B = [[7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
             [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
             [27, 28, 29, 30, 31, 32, 33, 34, 35, 36]]
        
        C = [[0.0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C[i][k] += A[i][j] * B[j][k]
        
        Cc = [[0.0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(rank):
                    ua = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                    vb = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                    acc += W[k*n+i][r] * ua * vb
                Cc[i][k] = acc
        
        for i in range(n):
            for k in range(p):
                if abs(C[i][k] - Cc[i][k]) > 1e-9:
                    return False
        return True

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook (rank 60)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    return {"U": U, "V": V, "W": W}
