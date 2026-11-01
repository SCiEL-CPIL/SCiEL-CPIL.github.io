"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    # Try rank-56 decomposition using Strassen on 8 blocks of 2 columns each
    # If that fails, fall back to schoolbook rank 64
    
    deadline = time.time() + 140  # safety margin
    
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
    
    def strassen_block_decomp():
        # 8 blocks of 2 columns each
        num_blocks = p // 2  # 8
        rank = num_blocks * 7  # 56
        U = [[0] * rank for _ in range(n * m)]  # 4 x 56
        V = [[0] * rank for _ in range(m * p)]  # 32 x 56
        W = [[0] * rank for _ in range(p * n)]  # 32 x 56
        
        # Strassen multipliers for 2x2:
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
        # W[0..31] = C[k,i]... wait, W[k*n+i] = W[k*2+i]
        # So W[2k] = C[0,k], W[2k+1] = C[1,k]
        
        for block in range(num_blocks):
            k0 = block * 2  # first column of this block
            k1 = k0 + 1     # second column of this block
            base = block * 7
            
            # M1 = (A00+A11)(B0k0+B1k1)
            # U: A00=1, A11=1; V: B0k0=1, B1k1=1
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
    
    def verify(decomp):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        n, m, p = DIMENSIONS
        # The tensor T[i,j,k] = delta[i,j,k] for matrix multiplication
        # T[i,j,k] = 1 if (i,j,k) is a valid index, and the product is A[i,j]*B[j,k]*C[i,k]
        # Actually, the tensor is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k]
        # where T[i,j,k] = 1 if this is a term in the matrix product, 0 otherwise
        # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k)
        # Wait, no. The tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j] B[j,k], so the tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
        # which means T[i,j,k] = 1 for all valid indices.
        
        # So we need: for all i, j, k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # And for invalid combinations (which don't exist in this case since all indices are valid), it should be 0.
        
        # Let's check: for each (i,j,k), compute the sum
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
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
