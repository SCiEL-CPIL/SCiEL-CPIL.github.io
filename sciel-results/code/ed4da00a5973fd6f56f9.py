import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def schoolbook():
        rank = n * m * p  # 64
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
        return {"U": U, "V": V, "W": W}
    
    def verify(decomp):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        n, m, p = DIMENSIONS
        # Check: for all i, j, k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Block-circulant / Kronecker product approach
    # Treat the 16 columns as 8 blocks of 2 columns each.
    # Use Strassen's algorithm on each 2x2 x 2x2 block (rank 7 per block).
    # Total rank = 8 * 7 = 56.
    
    def strassen_block_decomp():
        # 8 blocks of 2 columns each
        num_blocks = p // 2  # 8
        rank = num_blocks * 7  # 56
        U = [[0] * rank for _ in range(n * m)]  # 4 x 56
        V = [[0] * rank for _ in range(m * p)]  # 32 x 56
        W = [[0] * rank for _ in range(p * n)]  # 32 x 56
        
        # Strassen multipliers for 2x2 matrix multiplication:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 - M5 + M4 + M7
        # C01 = M5 + M3
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U indices: [i*2+j] for i in 0,1; j in 0,1
        # U[0] = A00, U[1] = A01, U[2] = A10, U[3] = A11
        
        # V indices: [j*16+k] for j in 0,1; k in 0..15
        # V[0..15] = B[0,k], V[16..31] = B[1,k]
        
        # W indices: [k*2+i] for k in 0..15; i in 0,1
        # W[2k] = C[0,k], W[2k+1] = C[1,k]
        
        for block in range(num_blocks):
            k0 = block * 2  # first column of this block
            k1 = k0 + 1     # second column of this block
            base = block * 7
            
            # M1 = (A00+A11)(B0k0+B1k1)
            U[0][base + 0] = 1   # A00
            U[3][base + 0] = 1   # A11
            V[0 * 16 + k0][base + 0] = 1   # B0k0
            V[1 * 16 + k1][base + 0] = 1   # B1k1
            # W: C0k0 += 1, C0k1 += 1, C1k0 += 1, C1k1 += 1
            W[k0 * 2 + 0][base + 0] = 1   # C0k0
            W[k1 * 2 + 0][base + 0] = 1   # C0k1
            W[k0 * 2 + 1][base + 0] = 1   # C1k0
            W[k1 * 2 + 1][base + 0] = 1   # C1k1
            
            # M2 = (A10+A11)B0k0
            U[2][base + 1] = 1   # A10
            U[3][base + 1] = 1   # A11
            V[0 * 16 + k0][base + 1] = 1   # B0k0
            # W: C1k0 += 1
            W[k0 * 2 + 1][base + 1] = 1   # C1k0
            
            # M3 = A00(B1k0-B1k1)
            U[0][base + 2] = 1   # A00
            V[1 * 16 + k0][base + 2] = 1   # B1k0
            V[1 * 16 + k1][base + 2] = -1  # B1k1
            # W: C0k1 += 1
            W[k1 * 2 + 0][base + 2] = 1   # C0k1
            
            # M4 = A11(B0k1-B0k0)
            U[3][base + 3] = 1   # A11
            V[0 * 16 + k1][base + 3] = 1   # B0k1
            V[0 * 16 + k0][base + 3] = -1  # B0k0
            # W: C0k0 += 1, C1k0 += 1
            W[k0 * 2 + 0][base + 3] = 1   # C0k0
            W[k0 * 2 + 1][base + 3] = 1   # C1k0
            
            # M5 = (A00+A01)B1k1
            U[0][base + 4] = 1   # A00
            U[1][base + 4] = 1   # A01
            V[1 * 16 + k1][base + 4] = 1   # B1k1
            # W: C0k0 -= 1, C0k1 += 1
            W[k0 * 2 + 0][base + 4] = -1  # C0k0
            W[k1 * 2 + 0][base + 4] = 1   # C0k1
            
            # M6 = (A10-A00)(B0k0+B0k1)
            U[2][base + 5] = 1   # A10
            U[0][base + 5] = -1  # A00
            V[0 * 16 + k0][base + 5] = 1   # B0k0
            V[0 * 16 + k1][base + 5] = 1   # B0k1
            # W: C1k1 += 1
            W[k1 * 2 + 1][base + 5] = 1   # C1k1
            
            # M7 = (A01-A11)(B1k0+B1k1)
            U[1][base + 6] = 1   # A01
            U[3][base + 6] = -1  # A11
            V[1 * 16 + k0][base + 6] = 1   # B1k0
            V[1 * 16 + k1][base + 6] = 1   # B1k1
            # W: C0k0 += 1
            W[k0 * 2 + 0][base + 6] = 1   # C0k0
        
        return {"U": U, "V": V, "W": W}
    
    # Try Strassen block decomposition
    try:
        decomp = strassen_block_decomp()
        if verify(decomp):
            return decomp
    except Exception:
        pass
    
    # Try a 4x4 block structure with 4 groups of 4 columns
    # Each 2x2 x 2x4 block has schoolbook rank 8
    # Total rank = 4 * 8 = 32
    def four_column_block_decomp():
        group_size = 4
        num_groups = p // group_size  # 4 groups
        rank = num_groups * n * m * group_size  # 4 * 2 * 2 * 4 = 64, wait that's schoolbook
        
        # Actually for 2x2 x 2x4, schoolbook is 2*2*4 = 8
        # So total rank = 4 * 8 = 32
        rank = num_groups * (n * m * group_size)  # 4 * 8 = 32
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        column = 0
        for g in range(num_groups):
            for i in range(n):
                for j in range(m):
                    for l in range(group_size):
                        col_idx = g * group_size + l
                        U[i * m + j][column] = 1
                        V[j * p + col_idx][column] = 1
                        W[col_idx * n + i][column] = 1
                        column += 1
        
        return {"U": U, "V": V, "W": W}
    
    try:
        decomp = four_column_block_decomp()
        if verify(decomp):
            return decomp
    except Exception:
        pass
    
    # Fall back to schoolbook
    return schoolbook()
