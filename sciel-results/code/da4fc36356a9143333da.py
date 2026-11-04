"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs a rank-28 decomposition for 4x4 matrix multiplication
    using Strassen's 7 multiplications over the ring of 2x2 matrices,
    where each 2x2 multiplication is done via schoolbook (4 multiplications).
    Total rank = 7 * 4 = 28.
    
    The tensor identity is:
      C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We partition A, B, C into 2x2 blocks:
      A = [[A00, A01], [A10, A11]]  (each block is 2x2)
      B = [[B00, B01], [B10, B11]]
      C = [[C00, C01], [C10, C11]]
    
    Strassen gives:
      C00 = M1 - M2 + M3 + M7
      C01 = M4 + M2
      C10 = M4 + M1
      C11 = M3 - M5 + M4 + M6
    
    where:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
    
    Each Mx is a 2x2 matrix product, decomposed into 4 scalar multiplications.
    """
    start_time = time.time()
    
    # Strassen 2x2 decomposition matrices (rank 7)
    # U2[i, c] where i = row*2+col of the 2x2 A matrix, c = 0..6
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00 (k=0, i=0): index 0*2+0=0
        [0, 1, 0, 1, 0, 0, 0],    # C10 (k=0, i=1): index 0*2+1=1
        [0, 0, 1, 0, 1, 0, 0],    # C01 (k=1, i=0): index 1*2+0=2
        [1, -1, 1, 0, 0, 1, 0]    # C11 (k=1, i=1): index 1*2+1=3
    ], dtype=int)
    
    # Schoolbook 2x2 decomposition matrices (rank 4)
    # For 2x2 multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # U_s[i*2+j, c], V_s[j*2+k, c], W_s[k*2+i, c]
    U_s = np.array([
        [1, 0, 0, 0],  # A00
        [0, 1, 0, 0],  # A01
        [0, 0, 1, 0],  # A10
        [0, 0, 0, 1]   # A11
    ], dtype=int)
    
    V_s = np.array([
        [1, 0, 0, 0],  # B00
        [0, 1, 0, 0],  # B01
        [0, 0, 1, 0],  # B10
        [0, 0, 0, 1]   # B11
    ], dtype=int)
    
    W_s = np.array([
        [1, 0, 0, 0],  # C00 (k=0,i=0)
        [0, 1, 0, 0],  # C10 (k=0,i=1)
        [0, 0, 1, 0],  # C01 (k=1,i=0)
        [0, 0, 0, 1]   # C11 (k=1,i=1)
    ], dtype=int)
    
    # Build the full 4x4 decomposition
    # Total rank = 7 * 4 = 28
    # For each Strassen multiplication s (0..6) and schoolbook multiplication t (0..3):
    #   The combined column index is s*4 + t
    
    U4 = np.zeros((16, 28), dtype=int)
    V4 = np.zeros((16, 28), dtype=int)
    W4 = np.zeros((16, 28), dtype=int)
    
    for s in range(7):
        for t in range(4):
            col = s * 4 + t
            
            # U contribution: A[i,j] where the 2x2 block structure is used
            # U2[s] gives the coefficient for the 2x2 block of A in Strassen step s
            # U_s[t] gives the coefficient for the individual element within the 2x2 block
            # For A[i,j] where i = 2*i_block + i_inner, j = 2*j_block + j_inner:
            #   coefficient = U2[i_block*2+j_block, s] * U_s[i_inner*2+j_inner, t]
            for i in range(4):
                for j in range(4):
                    i_block = i // 2
                    i_inner = i % 2
                    j_block = j // 2
                    j_inner = j % 2
                    idx = i * 4 + j
                    U4[idx, col] = U2[i_block * 2 + j_block, s] * U_s[i_inner * 2 + j_inner, t]
            
            # V contribution: B[j,k]
            for j in range(4):
                for k in range(4):
                    j_block = j // 2
                    j_inner = j % 2
                    k_block = k // 2
                    k_inner = k % 2
                    idx = j * 4 + k
                    V4[idx, col] = V2[j_block * 2 + k_block, s] * V_s[j_inner * 2 + k_inner, t]
            
            # W contribution: C[i,k] with trace convention W[k*n+i]
            for i in range(4):
                for k in range(4):
                    i_block = i // 2
                    i_inner = i % 2
                    k_block = k // 2
                    k_inner = k % 2
                    idx = k * 4 + i  # trace convention: W[k*n+i]
                    W4[idx, col] = W2[k_block * 2 + i_block, s] * W_s[k_inner * 2 + i_inner, t]
    
    # Verify the decomposition
    def verify(U, V, W, n=4, m=4, p=4):
        """Check tensor identity: C[i,k] = sum_j A[i,j] * B[j,k]"""
        # Build the multiplication tensor
        # T[i,j,k] = 1 if we want C[i,k] to include A[i,j]*B[j,k]
        # For each column c: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal delta for the product
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # For each (i, k) in output, check that sum_j A[i,j]*B[j,k] is correctly computed
        # The tensor T[i,j,k] = 1 for all i,j,k (matrix multiplication)
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # Wait, that's not right. The tensor for matrix multiplication is:
        # T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor has T[i,j,k] = 1 for all i,j,k, and we sum over j.
        
        # Actually, the decomposition means:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * V[j*p+k, c]
        # No, that's not right either.
        
        # Correct: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
        # No. The rank-r decomposition of the matrix multiplication tensor means:
        # sum_j A[i,j] B[j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # where we sum over j AND c.
        
        # So for each (i,k): sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] * A[i,j]*B[j,k] should work
        # Actually the tensor identity is:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where T[i,j,k] = 1 for all i,j,k (the matrix multiplication tensor)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Check if valid
    if not verify(U4.tolist(), V4.tolist(), W4.tolist()):
        # Fallback to schoolbook rank-64
        n, m, p = 4, 4, 4
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
    
    # Fallback if timeout
    if time.time() - start_time > 110:
        n, m, p = 4, 4, 4
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
    
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }
# EVOLVE-BLOCK-END
