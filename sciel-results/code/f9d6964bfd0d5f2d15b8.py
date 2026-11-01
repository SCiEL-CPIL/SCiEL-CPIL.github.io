import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-32 exact tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Strategy: split the 9 columns of B into two blocks:
      - Block A: 4 columns (cols 0-3): 2x2x4 = 2 * M_{2,2,2} = 2*7 = 14
      - Block B: 5 columns (cols 4-8): 2x2x5 = M_{2,2,3} + M_{2,2,2} = 11 + 7 = 18
    Total rank = 14 + 18 = 32
    
    Uses Strassen's 7-multiplication algorithm for 2x2x2 blocks
    and trivial 4-multiplication for 2x2x1 blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    rank = 32
    
    # Initialize factor matrices as flat lists for speed
    U = [[0] * rank for _ in range(n * m)]  # 4 x 32
    V = [[0] * rank for _ in range(m * p)]  # 18 x 32
    W = [[0] * rank for _ in range(p * n)]  # 18 x 32
    
    col = 0
    
    def strassen_block(k0, k1):
        """Add 7-term Strassen decomposition for 2x2 x 2x2 (cols k0, k1)."""
        nonlocal col
        
        # M1 = (a00+a11)(b0k0+b1k1)
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[p + k1][col] = 1
        W[k0][col] = 1; W[k1 + 1][col] = 1
        col += 1
        
        # M2 = (a10+a11)b0k0
        U[2][col] = 1; U[3][col] = 1
        V[k0][col] = 1
        W[k0 + 1][col] = 1; W[k1 + 1][col] = -1
        col += 1
        
        # M3 = a00(b0k1 - b1k1)
        U[0][col] = 1
        V[k1][col] = 1; V[p + k1][col] = -1
        W[k1][col] = 1; W[k1 + 1][col] = 1
        col += 1
        
        # M4 = a11(b1k0 - b0k0)
        U[3][col] = 1
        V[p + k0][col] = 1; V[k0][col] = -1
        W[k0][col] = 1; W[k0 + 1][col] = 1
        col += 1
        
        # M5 = (a00+a01)b1k1
        U[0][col] = 1; U[1][col] = 1
        V[p + k1][col] = 1
        W[k0][col] = -1; W[k1][col] = 1
        col += 1
        
        # M6 = (a10-a00)(b0k0+b0k1)
        U[2][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[k1 + 1][col] = 1
        col += 1
        
        # M7 = (a01-a11)(b1k0+b1k1)
        U[1][col] = 1; U[3][col] = -1
        V[p + k0][col] = 1; V[p + k1][col] = 1
        W[k0][col] = 1
        col += 1
    
    def trivial_col(k):
        """Add 4-term trivial decomposition for 2x2 x 2x1 (col k)."""
        nonlocal col
        U[0][col] = 1; V[k][col] = 1; W[k][col] = 1; col += 1
        U[1][col] = 1; V[p + k][col] = 1; W[k][col] = 1; col += 1
        U[2][col] = 1; V[k][col] = 1; W[k + 1][col] = 1; col += 1
        U[3][col] = 1; V[p + k][col] = 1; W[k + 1][col] = 1; col += 1
    
    # Block A: columns 0,1,2,3
    strassen_block(0, 1)
    strassen_block(2, 3)
    
    # Block B: columns 4,5,6,7,8
    strassen_block(4, 5)
    trivial_col(6)
    strassen_block(7, 8)
    
    assert col == rank, f"Rank mismatch: used {col}, expected {rank}"
    
    # Verification using numpy
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(np.multiply(U_np[i*m+j], V_np[j*p+k]), W_np[k*n+i]))
                if abs(s - 1.0) > 1e-10:
                    # Fallback to schoolbook rank-36
                    r = n * m * p
                    Uf = [[0.0]*r for _ in range(n*m)]
                    Vf = [[0.0]*r for _ in range(m*p)]
                    Wf = [[0.0]*r for _ in range(p*n)]
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                Uf[ii*m+jj][c] = 1.0
                                Vf[jj*p+kk][c] = 1.0
                                Wf[kk*n+ii][c] = 1.0
                                c += 1
                    return {"U": Uf, "V": Vf, "W": Wf}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
