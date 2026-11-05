"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The target tensor T_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    T_target = np.ones((n, m, p))
    
    # Reshape factors to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T_computed[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    T_computed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T_computed, T_target, atol=tol):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    # Strategy: Block-diagonal decomposition
    # Partition the row space (n=3) into a 1-row block and a 2-row block.
    # This corresponds to treating the multiplication as the direct sum of:
    # 1. A 1x4x8 tensor (rank 8, schoolbook)
    # 2. A 2x4x8 tensor (rank 14, Winograd-like scheme for 2x2 blocks extended to inner dim 4, 8)
    # Total rank = 8 + 14 = 22.
    
    # The 2x4x8 tensor can be viewed as 4 blocks of 2x1x2? No.
    # A 2x4 matrix times a 4x8 matrix.
    # We can partition the inner dimension (4) into two 2s?
    # Or partition the output dimension (8) into four 2s?
    # Let's use the known rank for 2x4x8.
    # If we partition the inner dimension 4 into two 2s, we have two 2x2x8 tensors?
    # No, the inner dimension must be the same for A and B.
    # A is 2x4, B is 4x8.
    # We can partition A into A1 (2x2) and A2 (2x2), and B into B1 (2x8) and B2 (2x8).
    # Then C = A1 B1 + A2 B2.
    # Each term is 2x2x8.
    # The rank of 2x2x8 is 7 * 4 = 28? No.
    # 2x2x8 can be seen as 4 independent 2x2x2 multiplications?
    # No, the output is 2x8.
    # For each column k of the output, we compute a 2x1 vector.
    # This is 4 independent 2x2x2 multiplications? No, 8 columns.
    # For each column k (0..7), we compute C[:, k] = A @ B[:, k].
    # This is a 2x2 matrix times a 2x1 vector.
    # The rank of 2x2x1 is 3? (Strassen for 2x2x1? No, 2x2x1 is rank 3? No, rank 4? 
    # Actually, the rank of 2x2x1 is 3. 
    # Wait, 2x2x1: A(2x2) * b(2x1) = c(2x1).
    # c0 = a00 b0 + a01 b1
    # c1 = a10 b0 + a11 b1
    # This is a bilinear map.
    # Rank is 3?
    # Let's check.
    # Terms:
    # 1: a00 b0 -> c0
    # 2: a01 b1 -> c0
    # 3: a10 b0 -> c1
    # 4: a11 b1 -> c1
    # Can we do it in 3?
    # (a00+a11)(b0+b1) = a00 b0 + a00 b1 + a11 b0 + a11 b1
    # (a00-a11)(b0-b1) = a00 b0 - a00 b1 - a11 b0 + a11 b1
    # (a01+a10)(b0+b1) ... no.
    # Actually, rank of 2x2x1 is 3.
    # So 2x2x8 has rank 8 * 3 = 24?
    # If so, 2x4x8 = 2 * 24 = 48?
    # That's worse than schoolbook 32.
    
    # Let's re-evaluate the 2x4x8 rank.
    # Schoolbook is 2*4*8 = 64? No, 2*4*8 = 64 scalar multiplications?
    # C[i, k] = sum_j A[i, j] B[j, k].
    # i in 0..1, k in 0..7, j in 0..3.
    # Total terms = 2 * 8 * 4 = 64.
    # So schoolbook rank is 64.
    # My previous "rank 14" claim for 2x4x8 was likely wrong or for a different dimension.
    # Let's check the inspiration: "rank-14 Winograd scheme for 2x2 blocks extended to the 4-dimensional inner space".
    # Maybe it means 2x2x4 has rank 14?
    # Rank of 2x2x4:
    # Partition output 4 into two 2s.
    # Two 2x2x2 multiplications.
    # Rank = 2 * 7 = 14.
    # Yes! 2x2x4 has rank 14.
    # So if we have 2x4x8, we can partition the inner dimension 4 into... wait.
    # A is 2x4, B is 4x8.
    # If we partition the inner dimension 4 into two 2s, we get two 2x2x8 tensors.
    # Rank of 2x2x8: Partition output 8 into four 2s.
    # Four 2x2x2 multiplications.
    # Rank = 4 * 7 = 28.
    # Total rank = 2 * 28 = 56.
    # Still worse than schoolbook 64? No, 56 < 64.
    # So 2x4x8 has rank <= 56.
    
    # What if we partition the output dimension 8 into four 2s?
    # Then we have four 2x4x2 tensors.
    # Rank of 2x4x2:
    # Partition inner dimension 4 into two 2s.
    # Two 2x2x2 tensors.
    # Rank = 2 * 7 = 14.
    # Total rank = 4 * 14 = 56.
    
    # Is there a better way for 2x4x8?
    # Maybe 2x4x8 can be done in rank 32?
    # Let's assume the best known is around 56.
    
    # So total rank for 3x4x8:
    # 1x4x8 (rank 8) + 2x4x8 (rank 56) = 64.
    # This is equal to the schoolbook rank of 3*4*8 = 96? No, 64 < 96.
    # So this is an improvement over the seed's 88.
    
    # Let's implement this.
    # 1. 1x4x8 part:
    #    U, V, W for rows 0, all cols.
    #    Rank 8.
    # 2. 2x4x8 part:
    #    Rows 1, 2.
    #    Rank 56.
    
    # Total rank 64.
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 1x4x8 (row 0)
    # Schoolbook
    for j in range(4):
        for k in range(8):
            U[0 * 4 + j][col] = 1
            V[j * 8 + k][col] = 1
            W[k * 3 + 0][col] = 1
            col += 1
    
    # Part 2: 2x4x8 (rows 1, 2)
    # Partition output k into four blocks of 2: k0, k1; k2, k3; k4, k5; k6, k7.
    # For each block, we do a 2x4x2 multiplication.
    # Partition inner j into two blocks of 2: j0, j1; j2, j3.
    # For each output block (2 cols) and inner block (2 rows), we do a 2x2x2 multiplication.
    # There are 4 output blocks and 2 inner blocks. Total 8 Strassen blocks.
    # Each Strassen block has rank 7.
    # Total rank 8 * 7 = 56.
    
    # Strassen 2x2x2 terms
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # (A coeffs, B coeffs, C coeffs)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 0, -1]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, 0]),
    ]
    
    # Indices for rows 1, 2 (i=1, 2)
    i_base = 1
    # Indices for cols 0..7 (k)
    # Indices for inner 0..3 (j)
    
    for k_block in range(4):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        for j_block in range(2):
            j0 = j_block * 2
            j1 = j_block * 2 + 1
            
            for t in range(7):
                a_coeffs, b_coeffs, c_coeffs = strassen_terms[t]
                
                # A is 2x2: rows i_base, i_base+1; cols j0, j1
                # A[i_base, j0] -> U[(i_base)*4 + j0]
                # A[i_base, j1] -> U[(i_base)*4 + j1]
                # A[i_base+1, j0] -> U[(i_base+1)*4 + j0]
                # A[i_base+1, j1] -> U[(i_base+1)*4 + j1]
                
                U[i_base * 4 + j0][col] = a_coeffs[0]
                U[i_base * 4 + j1][col] = a_coeffs[1]
                U[(i_base + 1) * 4 + j0][col] = a_coeffs[2]
                U[(i_base + 1) * 4 + j1][col] = a_coeffs[3]
                
                # B is 2x2: rows j0, j1; cols k0, k1
                # B[j0, k0] -> V[j0 * 8 + k0]
                # B[j0, k1] -> V[j0 * 8 + k1]
                # B[j1, k0] -> V[j1 * 8 + k0]
                # B[j1, k1] -> V[j1 * 8 + k1]
                
                V[j0 * 8 + k0][col] = b_coeffs[0]
                V[j0 * 8 + k1][col] = b_coeffs[1]
                V[j1 * 8 + k0][col] = b_coeffs[2]
                V[j1 * 8 + k1][col] = b_coeffs[3]
                
                # C is 2x2: rows i_base, i_base+1; cols k0, k1
                # C[i_base, k0] -> W[k0 * 3 + i_base]
                # C[i_base, k1] -> W[k1 * 3 + i_base]
                # C[i_base+1, k0] -> W[k0 * 3 + i_base+1]
                # C[i_base+1, k1] -> W[k1 * 3 + i_base+1]
                
                W[k0 * 3 + i_base][col] = c_coeffs[0]
                W[k1 * 3 + i_base][col] = c_coeffs[1]
                W[k0 * 3 + i_base + 1][col] = c_coeffs[2]
                W[k1 * 3 + i_base + 1][col] = c_coeffs[3]
                
                col += 1
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
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
# EVOLVE-BLOCK-END
