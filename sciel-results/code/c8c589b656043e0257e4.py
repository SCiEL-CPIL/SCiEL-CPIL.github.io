import numpy as np
import time

def construct():
    n, m, p = 2, 2, 9
    start_time = time.time()
    deadline = start_time + 145
    
    # Rank 32 decomposition using Strassen for 4x(2x2) blocks and schoolbook for 1x(2x1) block
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for 2x2 * 2x2
    # P1 = (a+d)(B00+B11)
    # P2 = (c+d)B00
    # P3 = a(B01-B11)
    # P4 = d(B10-B00)
    # P5 = (a+b)B11
    # P6 = (c-a)(B00+B01)
    # P7 = (b-d)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 - P6
    
    # Let's verify these formulas with a quick mental check or code.
    # I will add a verification step in the code.
    
    def add_strassen_block(k1, k2, col_offset):
        nonlocal col
        # P1
        U[0][col_offset] = 1; U[3][col_offset] = 1
        V[0*p+k1][col_offset] = 1; V[1*p+k2][col_offset] = 1
        W[k1*n+0][col_offset] = 1; W[k2*n+0][col_offset] = 0
        W[k1*n+1][col_offset] = 0; W[k2*n+1][col_offset] = 1
        col += 1
        
        # P2
        U[2][col_offset] = 1; U[3][col_offset] = 1
        V[0*p+k1][col_offset] = 1
        W[k1*n+0][col_offset] = 0; W[k2*n+0][col_offset] = 0
        W[k1*n+1][col_offset] = 1; W[k2*n+1][col_offset] = 0
        col += 1
        
        # P3
        U[0][col_offset] = 1
        V[0*p+k2][col_offset] = 1; V[1*p+k2][col_offset] = -1
        W[k1*n+0][col_offset] = 0; W[k2*n+0][col_offset] = 1
        W[k1*n+1][col_offset] = 0; W[k2*n+1][col_offset] = 0
        col += 1
        
        # P4
        U[3][col_offset] = 1
        V[1*p+k1][col_offset] = 1; V[0*p+k1][col_offset] = -1
        W[k1*n+0][col_offset] = 1; W[k2*n+0][col_offset] = 0
        W[k1*n+1][col_offset] = 1; W[k2*n+1][col_offset] = 0
        col += 1
        
        # P5
        U[0][col_offset] = 1; U[1][col_offset] = 1
        V[1*p+k2][col_offset] = 1
        W[k1*n+0][col_offset] = -1; W[k2*n+0][col_offset] = 1
        W[k1*n+1][col_offset] = 0; W[k2*n+1][col_offset] = 0
        col += 1
        
        # P6
        U[2][col_offset] = 1; U[0][col_offset] = -1
        V[0*p+k1][col_offset] = 1; V[0*p+k2][col_offset] = 1
        W[k1*n+0][col_offset] = 0; W[k2*n+0][col_offset] = 0
        W[k1*n+1][col_offset] = 0; W[k2*n+1][col_offset] = -1
        col += 1
        
        # P7
        U[1][col_offset] = 1; U[3][col_offset] = -1
        V[1*p+k1][col_offset] = 1; V[1*p+k2][col_offset] = 1
        W[k1*n+0][col_offset] = 1; W[k2*n+0][col_offset] = 0
        W[k1*n+1][col_offset] = 0; W[k2*n+1][col_offset] = 0
        col += 1

    # Apply to 4 blocks
    for i in range(4):
        k1 = i * 2
        k2 = i * 2 + 1
        add_strassen_block(k1, k2, col)
        
    # Last column k=8
    k = 8
    # Schoolbook 2x2 * 2x1
    # C0 = a*B0 + b*B1
    # C1 = c*B0 + d*B1
    
    # a*B0
    U[0][col] = 1
    V[0*p+k][col] = 1
    W[k*n+0][col] = 1
    col += 1
    
    # b*B1
    U[1][col] = 1
    V[1*p+k][col] = 1
    W[k*n+0][col] = 1
    col += 1
    
    # c*B0
    U[2][col] = 1
    V[0*p+k][col] = 1
    W[k*n+1][col] = 1
    col += 1
    
    # d*B1
    U[3][col] = 1
    V[1*p+k][col] = 1
    W[k*n+1][col] = 1
    col += 1
    
    assert col == rank
    
    # Verification
    def verify(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    u_val = sum(U[i*m+j, r] * A[i,j] for j in range(m))
                    v_val = sum(V[j*p+k, r] * B[j,k] for j in range(m))
                    C_decomp[i,k] += W[k*n+i, r] * u_val * v_val
        return np.allclose(C_expected, C_decomp, atol=1e-10)
        
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook 36
        rank = 36
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    assert col <= rank
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
