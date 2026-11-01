import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p  # 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # u = i*m + j
                    # v = j*p + k
                    # w = k*n + i
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen block decomposition
    def strassen_block_decomp():
        # 8 blocks of 2 columns each
        num_blocks = p // 2  # 8
        rank = num_blocks * 7  # 56
        U = [[0] * rank for _ in range(n * m)]  # 4 x 56
        V = [[0] * rank for _ in range(m * p)]  # 32 x 56
        W = [[0] * rank for _ in range(p * n)]  # 32 x 56
        
        for block in range(num_blocks):
            k0 = block * 2  # first column of this block
            k1 = k0 + 1     # second column of this block
            base = block * 7
            
            # u indices: 0:A00, 1:A01, 2:A10, 3:A11
            # v indices: j*p + k
            # w indices: k*n + i = 2k + i
            
            # Term 0: M1 = (A00+A11)(B0k0+B1k1)
            # U: A00(1), A11(1)
            U[0][base + 0] = 1
            U[3][base + 0] = 1
            # V: B0k0(1), B1k1(1)
            V[0 * p + k0][base + 0] = 1
            V[1 * p + k1][base + 0] = 1
            # W: C0k0(1), C1k1(1) -> w=2k0+0, w=2k1+1
            W[k0 * n + 0][base + 0] = 1
            W[k1 * n + 1][base + 0] = 1
            
            # Term 1: M2 = (A10+A11)B0k0
            # U: A10(1), A11(1)
            U[2][base + 1] = 1
            U[3][base + 1] = 1
            # V: B0k0(1)
            V[0 * p + k0][base + 1] = 1
            # W: C1k0(1), C1k1(-1) -> w=2k0+1, w=2k1+1
            W[k0 * n + 1][base + 1] = 1
            W[k1 * n + 1][base + 1] = -1
            
            # Term 2: M3 = A00(B1k0-B1k1)
            # U: A00(1)
            U[0][base + 2] = 1
            # V: B1k0(1), B1k1(-1)
            V[1 * p + k0][base + 2] = 1
            V[1 * p + k1][base + 2] = -1
            # W: C0k1(1), C1k1(1) -> w=2k1+0, w=2k1+1
            W[k1 * n + 0][base + 2] = 1
            W[k1 * n + 1][base + 2] = 1
            
            # Term 3: M4 = A11(B0k1-B0k0)
            # U: A11(1)
            U[3][base + 3] = 1
            # V: B0k1(1), B0k0(-1)
            V[0 * p + k1][base + 3] = 1
            V[0 * p + k0][base + 3] = -1
            # W: C0k0(1), C1k0(1) -> w=2k0+0, w=2k0+1
            W[k0 * n + 0][base + 3] = 1
            W[k0 * n + 1][base + 3] = 1
            
            # Term 4: M5 = (A00+A01)B1k1
            # U: A00(1), A01(1)
            U[0][base + 4] = 1
            U[1][base + 4] = 1
            # V: B1k1(1)
            V[1 * p + k1][base + 4] = 1
            # W: C0k0(-1), C0k1(1) -> w=2k0+0, w=2k1+0
            W[k0 * n + 0][base + 4] = -1
            W[k1 * n + 0][base + 4] = 1
            
            # Term 5: M6 = (A10-A00)(B0k0+B0k1)
            # U: A10(1), A00(-1)
            U[2][base + 5] = 1
            U[0][base + 5] = -1
            # V: B0k0(1), B0k1(1)
            V[0 * p + k0][base + 5] = 1
            V[0 * p + k1][base + 5] = 1
            # W: C1k1(1) -> w=2k1+1
            W[k1 * n + 1][base + 5] = 1
            
            # Term 6: M7 = (A01-A11)(B1k0+B1k1)
            # U: A01(1), A11(-1)
            U[1][base + 6] = 1
            U[3][base + 6] = -1
            # V: B1k0(1), B1k1(1)
            V[1 * p + k0][base + 6] = 1
            V[1 * p + k1][base + 6] = 1
            # W: C0k0(1) -> w=2k0+0
            W[k0 * n + 0][base + 6] = 1
            
        return {"U": U, "V": V, "W": W}
    
    def verify(decomp):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        
        # Check shapes
        if U.shape != (n*m, decomp["U"][0].__len__()) or V.shape != (m*p, decomp["V"][0].__len__()) or W.shape != (p*n, decomp["W"][0].__len__()):
             return False
             
        rank = U.shape[1]
        # For each i, j, k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        # We can vectorize this check for speed, but n,m,p are small.
        # Let's just loop to be safe and simple, it's only 2*2*16 = 64 checks.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try Strassen block decomposition
    try:
        decomp = strassen_block_decomp()
        if verify(decomp):
            return decomp
    except Exception:
        pass
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
