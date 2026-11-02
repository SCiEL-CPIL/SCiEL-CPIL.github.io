import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 exact tensor decomposition for 2x2 * 2x12 matrix multiplication.
    
    Uses Strassen's 7-product algorithm on 6 independent 2x2x2 blocks.
    Total rank = 6 * 7 = 42.
    
    Indexing convention:
      U[i*m+j, r]: coefficient for A[i][j]
      V[j*p+k, r]: coefficient for B[j][k]  
      W[k*n+i, r]: coefficient for C[i][k]
    """
    n, m, p = DIMENSIONS  # 2, 2, 12
    t_start = time.time()
    deadline = t_start + 110
    
    num_blocks = p // 2  # 6 blocks of 2 columns each
    rank_per_block = 7
    rank = num_blocks * rank_per_block  # 42
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen 7-product decomposition for 2x2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for t in range(num_blocks):
        base = t * rank_per_block
        
        # B indices for this block: columns 2t and 2t+1
        b00 = 0 * p + 2 * t      # B[0, 2t]
        b01 = 0 * p + 2 * t + 1  # B[0, 2t+1]
        b10 = 1 * p + 2 * t      # B[1, 2t]
        b11 = 1 * p + 2 * t + 1  # B[1, 2t+1]
        
        # C indices for this block (W convention: k*n + i)
        c00 = (2 * t) * n + 0    # C[0, 2t]
        c01 = (2 * t + 1) * n + 0  # C[0, 2t+1]
        c10 = (2 * t) * n + 1    # C[1, 2t]
        c11 = (2 * t + 1) * n + 1  # C[1, 2t+1]
        
        # A indices
        a00 = 0 * m + 0  # A[0,0] = a
        a01 = 0 * m + 1  # A[0,1] = b
        a10 = 1 * m + 0  # A[1,0] = c
        a11 = 1 * m + 1  # A[1,1] = d
        
        # M1 = (a+d)(e+h)
        r = base + 0
        U[a00][r] = 1
        U[a11][r] = 1
        V[b00][r] = 1
        V[b11][r] = 1
        W[c00][r] = 1
        W[c11][r] = 1
        
        # M2 = (c+d)e
        r = base + 1
        U[a10][r] = 1
        U[a11][r] = 1
        V[b00][r] = 1
        W[c10][r] = 1
        W[c11][r] = -1
        
        # M3 = a(f-h)
        r = base + 2
        U[a00][r] = 1
        V[b01][r] = 1
        V[b11][r] = -1
        W[c01][r] = 1
        W[c11][r] = 1
        
        # M4 = d(g-e)
        r = base + 3
        U[a11][r] = 1
        V[b10][r] = 1
        V[b00][r] = -1
        W[c00][r] = 1
        W[c10][r] = 1
        
        # M5 = (a+b)h
        r = base + 4
        U[a00][r] = 1
        U[a01][r] = 1
        V[b11][r] = 1
        W[c00][r] = -1
        W[c01][r] = 1
        
        # M6 = (c-a)(e+f)
        r = base + 5
        U[a10][r] = 1
        U[a00][r] = -1
        V[b00][r] = 1
        V[b01][r] = 1
        W[c11][r] = 1
        
        # M7 = (b-d)(g+h)
        r = base + 6
        U[a01][r] = 1
        U[a11][r] = -1
        V[b10][r] = 1
        V[b11][r] = 1
        W[c00][r] = 1
    
    # Verify the decomposition
    if time.time() < deadline - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            # Fallback to schoolbook rank 48
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n * m)]
            V_s = [[0] * rank_school for _ in range(m * p)]
            W_s = [[0] * rank_school for _ in range(p * n)]
            c = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_s[ii*m+jj][c] = 1
                        V_s[jj*p+kk][c] = 1
                        W_s[kk*n+ii][c] = 1
                        c += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
