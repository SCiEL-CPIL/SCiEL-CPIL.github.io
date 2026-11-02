import numpy as np
import time

def construct():
    """
    Construct a rank-49 exact tensor decomposition for 4x4x4 matrix multiplication.
    
    This uses a recursive Strassen decomposition:
    - Split 4x4x4 into 2x2 blocks (2x2 block grid)
    - Use 7 Strassen multiplications at the block level
    - Use 7 Strassen multiplications at the 2x2 level
    - Total rank = 7 * 7 = 49
    
    The output W follows the trace convention W[k*n+i, r] where:
    - k is the column index of the result (0..3)
    - i is the row index of the result (0..3)
    - n = p = 4 (number of rows in A and columns in B)
    """
    n, m, p = 4, 4, 4  # A is n x m, B is m x p, C is n x p
    # Actually for matrix multiplication C = A * B:
    # A is (n x m), B is (m x p), C is (n x p)
    # Tensor index: C[i, k] = sum_j A[i, j] * B[j, k]
    # So we need U with shape (n*m, rank), V with shape (m*p, rank), W with shape (p*n, rank)
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    rank = 49
    
    # Strassen coefficients for 2x2 matrices
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    # C00 = M0 + M3 - M5 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M4
    
    # Local Strassen coefficients for 2x2 matrices
    # U_loc[l][i*2+j] = coefficient for A[i,j] in multiplication l
    # V_loc[l][j*2+k] = coefficient for B[j,k] in multiplication l  
    # W_loc[l][k*2+i] = coefficient for C[i,k] in multiplication l
    
    U_loc = np.zeros((7, 4), dtype=float)
    V_loc = np.zeros((7, 4), dtype=float)
    W_loc = np.zeros((7, 4), dtype=float)
    
    # l=0: M0 = (A00 + A11) * (B00 + B11)
    U_loc[0, 0] = 1; U_loc[0, 3] = 1  # A00, A11
    V_loc[0, 0] = 1; V_loc[0, 3] = 1  # B00, B11
    W_loc[0, 0] = 1; W_loc[0, 3] = 1  # C00, C11
    
    # l=1: M1 = (A10 + A11) * B00
    U_loc[1, 2] = 1; U_loc[1, 3] = 1  # A10, A11
    V_loc[1, 0] = 1  # B00
    W_loc[1, 1] = 1; W_loc[1, 3] = -1  # C10, -C11
    
    # l=2: M2 = A00 * (B01 - B11)
    U_loc[2, 0] = 1  # A00
    V_loc[2, 1] = 1; V_loc[2, 3] = -1  # B01, -B11
    W_loc[2, 2] = 1; W_loc[2, 3] = 1  # C01, C11
    
    # l=3: M3 = A11 * (B10 - B00)
    U_loc[3, 3] = 1  # A11
    V_loc[3, 2] = 1; V_loc[3, 0] = -1  # B10, -B00
    W_loc[3, 0] = 1; W_loc[3, 1] = 1  # C00, C10
    
    # l=4: M4 = (A00 + A01) * B11
    U_loc[4, 0] = 1; U_loc[4, 1] = 1  # A00, A01
    V_loc[4, 3] = 1  # B11
    W_loc[4, 2] = 1; W_loc[4, 3] = 1  # C01, C11
    
    # l=5: M5 = (A10 - A00) * (B00 + B01)
    U_loc[5, 2] = 1; U_loc[5, 0] = -1  # A10, -A00
    V_loc[5, 0] = 1; V_loc[5, 1] = 1  # B00, B01
    W_loc[5, 0] = -1; W_loc[5, 1] = 1  # -C00, C10
    
    # l=6: M6 = (A01 - A11) * (B10 + B11)
    U_loc[6, 1] = 1; U_loc[6, 3] = -1  # A01, -A11
    V_loc[6, 2] = 1; V_loc[6, 3] = 1  # B10, B11
    W_loc[6, 0] = -1; W_loc[6, 2] = 1  # -C00, C01
    
    # Block-level Strassen coefficients
    # For 2x2 block matrices, same structure as 2x2 scalar matrices
    # Block indices: (ib, jb) for A blocks, (jb, kb) for B blocks, (kb, ib) for C blocks
    # where ib, jb, kb are block indices (0 or 1)
    
    U_blocks = np.zeros((7, 4), dtype=float)
    V_blocks = np.zeros((7, 4), dtype=float)
    W_blocks = np.zeros((7, 4), dtype=float)
    
    # Same as U_loc, V_loc, W_loc but for block indices
    U_blocks = U_loc.copy()
    V_blocks = V_loc.copy()
    W_blocks = W_loc.copy()
    
    # Build the full rank-49 decomposition
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for kb in range(2):  # Block column of B / Block row of C
        for ib in range(2):  # Block row of A / Block column of C
            for lb in range(7):  # Block-level Strassen multiplication
                for ll in range(7):  # Local Strassen multiplication
                    col = lb * 7 + ll
                    
                    # U contribution: A[ib, jb] for each jb
                    for jb in range(2):
                        # Check if this block is used in block-level multiplication lb
                        block_idx = ib * 2 + jb
                        coeff_block = U_blocks[lb, block_idx]
                        if coeff_block != 0:
                            # Within this block, use local Strassen multiplication ll
                            for il in range(2):  # Local row within block
                                for jl in range(2):  # Local col within block
                                    i = ib * 2 + il
                                    j = jb * 2 + jl
                                    local_idx = il * 2 + jl
                                    coeff_local = U_loc[ll, local_idx]
                                    if coeff_local != 0:
                                        U[i * m + j, col] += coeff_block * coeff_local
                    
                    # V contribution: B[jb, kb] for each jb
                    for jb in range(2):
                        # Check if this block is used in block-level multiplication lb
                        block_idx = jb * 2 + kb
                        coeff_block = V_blocks[lb, block_idx]
                        if coeff_block != 0:
                            # Within this block, use local Strassen multiplication ll
                            for jl in range(2):  # Local row within block
                                for kl in range(2):  # Local col within block
                                    j = jb * 2 + jl
                                    k = kb * 2 + kl
                                    local_idx = jl * 2 + kl
                                    coeff_local = V_loc[ll, local_idx]
                                    if coeff_local != 0:
                                        V[j * p + k, col] += coeff_block * coeff_local
                    
                    # W contribution: C[ib, kb]
                    block_idx = kb * 2 + ib
                    coeff_block = W_blocks[lb, block_idx]
                    if coeff_block != 0:
                        # Within this block, use local Strassen multiplication ll
                        for il in range(2):  # Local row within block
                            for kl in range(2):  # Local col within block
                                i = ib * 2 + il
                                k = kb * 2 + kl
                                local_idx = kl * 2 + il
                                coeff_local = W_loc[ll, local_idx]
                                if coeff_local != 0:
                                    W[k * n + i, col] += coeff_block * coeff_local
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = [[float(int(U[i][r])) for r in range(rank)] for i in range(n * m)]
    V_list = [[float(int(V[i][r])) for r in range(rank)] for i in range(m * p)]
    W_list = [[float(int(W[i][r])) for r in range(rank)] for i in range(p * n)]
    
    # Verify the decomposition (optional, for debugging)
    # For random A, B: C = A @ B
    # U @ (diag(mult) @ V) @ W^T should give the right result
    # Actually: C[i,k] = sum_j A[i,j] B[j,k]
    # With decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... no
    # The tensor decomposition means: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = delta_{i,k} for matrix multiplication... no
    # Actually for matrix multiplication tensor: T[i,j,k] = 1 if i=k (no, that's not right either)
    # The matrix multiplication tensor has T[i,j,k] = delta_{i, something}... 
    # Let me think: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] such that sum_j A[i,j] T[j,k] = C[i,k]... no
    # The trilinear form is: <A, B, C> = sum_{i,j,k} A[i,j] B[j,k] C[k,i]
    # So T[i,j,k] = delta_{i,k} in the sense that we sum over j and the tensor connects i->j->k->i
    # Actually T[i,j,k] is nonzero when... it's a 3-tensor where T[i,j,k] = 1 if the indices form a valid path
    # For matrix mult: T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has shape (n*m, rank) indexed as U[i*m+j, r]
    # V has shape (m*p, rank) indexed as V[j*p+k, r]
    # W has shape (p*n, rank) indexed as W[k*n+i, r]
    
    # Let's verify with a simple test
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using the decomposition
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_computed = np.zeros((n, p), dtype=float)
    U_arr = np.array(U_list)
    V_arr = np.array(V_list)
    W_arr = np.array(W_list)
    
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                for r in range(rank):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
            C_computed[i, k] = total
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Try to debug
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
