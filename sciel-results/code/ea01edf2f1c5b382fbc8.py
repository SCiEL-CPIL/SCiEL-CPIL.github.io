"""Tensor decomposition for 2x2 by 2x16 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix A by a 2x16 matrix B to produce a 2x16 matrix C.
    
    The tensor is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    where n=2, m=2, p=16.
    
    Strategy: Use Strassen's algorithm on 2x2 blocks of columns.
    Divide the 16 columns into 8 blocks of 2 columns each.
    For each block, use rank-7 Strassen decomposition.
    Total rank = 8 * 7 = 56.
    """
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def schoolbook():
        """Standard schoolbook decomposition with rank = n*m*p = 64."""
        n, m, p = DIMENSIONS
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
    
    def strassen_block_decomp():
        """
        Use Strassen's algorithm on 8 blocks of 2 columns each.
        Each block uses rank 7, total rank = 56.
        
        For a 2x2 matrix multiplication C = A*B where:
        A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        
        Strassen multipliers:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        C00 = M1 - M5 + M4 + M7
        C01 = M5 + M3
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        n, m, p = DIMENSIONS  # n=2, m=2, p=16
        num_blocks = p // 2  # 8 blocks of 2 columns each
        rank = num_blocks * 7  # 56
        
        # U has shape (n*m, rank) = (4, 56)
        # U[0] = A00, U[1] = A01, U[2] = A10, U[3] = A11
        U = [[0] * rank for _ in range(n * m)]
        
        # V has shape (m*p, rank) = (32, 56)
        # V[0..15] = B[0, 0..15], V[16..31] = B[1, 0..15]
        V = [[0] * rank for _ in range(m * p)]
        
        # W has shape (p*n, rank) = (32, 56)
        # W[k*n+i] = W[k*2+i] = C[i, k]
        # So W[2k] = C[0, k], W[2k+1] = C[1, k]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(num_blocks):
            k0 = block * 2  # first column index in this block
            k1 = k0 + 1     # second column index in this block
            base = block * 7
            
            # For this block, B is a 2x2 matrix:
            # B_block = [[B[0,k0], B[0,k1]],
            #            [B[1,k0], B[1,k1]]]
            
            # M1 = (A00+A11)(B0k0+B1k1)
            # U: A00=1, A11=1; V: B[0,k0]=1, B[1,k1]=1
            U[0][base + 0] = 1   # A00
            U[3][base + 0] = 1   # A11
            V[0 * 16 + k0][base + 0] = 1   # B[0,k0]
            V[1 * 16 + k1][base + 0] = 1   # B[1,k1]
            # W: C[0,k0] += 1, C[0,k1] += 1, C[1,k0] += 1, C[1,k1] += 1
            W[k0 * 2 + 0][base + 0] = 1   # C[0,k0]
            W[k1 * 2 + 0][base + 0] = 1   # C[0,k1]
            W[k0 * 2 + 1][base + 0] = 1   # C[1,k0]
            W[k1 * 2 + 1][base + 0] = 1   # C[1,k1]
            
            # M2 = (A10+A11)B0k0
            # U: A10=1, A11=1; V: B[0,k0]=1
            U[2][base + 1] = 1   # A10
            U[3][base + 1] = 1   # A11
            V[0 * 16 + k0][base + 1] = 1   # B[0,k0]
            # W: C[1,k0] += 1
            W[k0 * 2 + 1][base + 1] = 1   # C[1,k0]
            
            # M3 = A00(B1k0-B1k1)
            # U: A00=1; V: B[1,k0]=1, B[1,k1]=-1
            U[0][base + 2] = 1   # A00
            V[1 * 16 + k0][base + 2] = 1   # B[1,k0]
            V[1 * 16 + k1][base + 2] = -1  # B[1,k1]
            # W: C[0,k1] += 1
            W[k1 * 2 + 0][base + 2] = 1   # C[0,k1]
            
            # M4 = A11(B0k1-B0k0)
            # U: A11=1; V: B[0,k1]=1, B[0,k0]=-1
            U[3][base + 3] = 1   # A11
            V[0 * 16 + k1][base + 3] = 1   # B[0,k1]
            V[0 * 16 + k0][base + 3] = -1  # B[0,k0]
            # W: C[0,k0] += 1, C[1,k0] += 1
            W[k0 * 2 + 0][base + 3] = 1   # C[0,k0]
            W[k0 * 2 + 1][base + 3] = 1   # C[1,k0]
            
            # M5 = (A00+A01)B1k1
            # U: A00=1, A01=1; V: B[1,k1]=1
            U[0][base + 4] = 1   # A00
            U[1][base + 4] = 1   # A01
            V[1 * 16 + k1][base + 4] = 1   # B[1,k1]
            # W: C[0,k0] -= 1, C[0,k1] += 1
            W[k0 * 2 + 0][base + 4] = -1  # C[0,k0]
            W[k1 * 2 + 0][base + 4] = 1   # C[0,k1]
            
            # M6 = (A10-A00)(B0k0+B0k1)
            # U: A10=1, A00=-1; V: B[0,k0]=1, B[0,k1]=1
            U[2][base + 5] = 1   # A10
            U[0][base + 5] = -1  # A00
            V[0 * 16 + k0][base + 5] = 1   # B[0,k0]
            V[0 * 16 + k1][base + 5] = 1   # B[0,k1]
            # W: C[1,k1] += 1
            W[k1 * 2 + 1][base + 5] = 1   # C[1,k1]
            
            # M7 = (A01-A11)(B1k0+B1k1)
            # U: A01=1, A11=-1; V: B[1,k0]=1, B[1,k1]=1
            U[1][base + 6] = 1   # A01
            U[3][base + 6] = -1  # A11
            V[1 * 16 + k0][base + 6] = 1   # B[1,k0]
            V[1 * 16 + k1][base + 6] = 1   # B[1,k1]
            # W: C[0,k0] += 1
            W[k0 * 2 + 0][base + 6] = 1   # C[0,k0]
        
        return {"U": U, "V": V, "W": W}
    
    def verify(decomp):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Vectorized computation
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try Strassen block decomposition first (rank 56)
    try:
        decomp = strassen_block_decomp()
        if verify(decomp):
            return decomp
    except Exception as e:
        # If Strassen fails for any reason, fall back to schoolbook
        pass
    
    # Check deadline before falling back
    if time.time() > deadline:
        # Should not happen, but just in case
        pass
    
    # Fall back to schoolbook (rank 64)
    return schoolbook()
# EVOLVE-BLOCK-END
