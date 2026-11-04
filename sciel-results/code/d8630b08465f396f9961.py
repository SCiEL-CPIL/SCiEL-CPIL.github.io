"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    using the tensor product of the rank-7 Strassen decomposition for 2x2.
    Uses NumPy kron with proper permutation to map to row-major linear indices.
    """
    start_time = time.time()
    
    # Define the 2x2 Strassen decomposition matrices
    # U2[i, c] where i = row*2+col of the 2x2 A matrix, c = 0..6
    # V2[j, c] where j = row*2+col of the 2x2 B matrix, c = 0..6  
    # W2[k, c] where k = col*2+row of the 2x2 C matrix (trace convention: W[k*n+i]), c = 0..6
    
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
    
    # Kronecker product for 4x4
    # U4 = kron(U2, U2), V4 = kron(V2, V2), W4 = kron(W2, W2)
    # np.kron(A, B) has shape (A.rows*B.rows, A.cols*B.cols)
    # Row index in kron: R = i_A * B.rows + i_B
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Permutation to map Kronecker row indices to standard row-major linear indices
    # For 4x4 matrix: linear index S = r*4 + c
    # Block decomposition: r = 2*r_out + r_in, c = 2*c_out + c_in
    # Kronecker index: L4 = L2_out * 4 + L2_in
    #   where L2_out = r_out*2 + c_out, L2_in = r_in*2 + c_in
    
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_out = r // 2
        r_in = r % 2
        c_out = c // 2
        c_in = c % 2
        
        L2_out = r_out * 2 + c_out
        L2_in = r_in * 2 + c_in
        
        L4 = L2_out * 4 + L2_in
        perm[S] = L4
    
    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]
    
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
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
# EVOLVE-BLOCK-END
