import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    _start_time = time.time()
    _deadline = 140  # 10s safety margin before 150s kill
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    # Indices:
    # U[i, k] where i in 0..n-1, k in 0..m-1. Index = i*m + k
    # V[k, j] where k in 0..m-1, j in 0..p-1. Index = k*p + j
    # W[j, i] where j in 0..p-1, i in 0..n-1. Index = j*n + i (Trace convention: W[k*n+i] in prompt? 
    # Prompt says W[k*n+i] for W. Let's re-read carefully.
    # "output factor follows the official trace convention W[k*n+i]"
    # In the seed: W[k*n+i] where k is the column of B (0..p-1) and i is row of A (0..n-1).
    # So W index is j*n + i where j is col of B (0..p-1) and i is row of A (0..n-1).
    # This corresponds to C[i, j].
    
    # Target Tensor T[a, b, c]
    # a = i*m + k
    # b = k*p + j
    # c = j*n + i
    # T[a, b, c] = 1 if i,k,j match.
    
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify_solution(U, V, W):
        # Check if decomposition is exact
        D = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(D, T, atol=1e-9):
            return False
        # Check integer/half-integer constraint
        for mat in [U, V, W]:
            if np.any(np.abs(mat * 2 - np.round(mat * 2)) > 1e-9):
                return False
        return True

    # Strategy:
    # 1. Start with schoolbook (rank 48).
    # 2. Try to reduce rank by 1 using "rank reduction" techniques (e.g., if a column is a linear combination of others, or using Strassen-like tricks).
    # 3. Specifically, look for rank 47, 46...
    # 
    # A known result for 3x4x4 matrix multiplication is rank 44 (using Strassen on 2x2x2 blocks).
    # Let's try to construct a rank 44 solution explicitly.
    
    # Construction of Rank 44 Solution:
    # Split A (3x4) into A1 (2x2) and A2 (2x2) vertically? No, A is 3 rows.
    # Split A into A_top (2x4) and A_bot (1x4).
    # Split B (4x4) into B_left (4x2) and B_right (4x2).
    # 
    # C = A * B = [ A_top * B_left, A_top * B_right ]
    #             [ A_bot * B_left, A_bot * B_right ]
    # 
    # A_top * B_left: 2x2 * 4x2. This is not a standard block multiplication.
    # Let's use a different split.
    # A (3x4) -> A1 (2x2) (top-left), A2 (2x2) (top-right), A3 (1x2) (bot-left), A4 (1x2) (bot-right).
    # B (4x4) -> B1 (2x2) (top-left), B2 (2x2) (top-right), B3 (2x2) (bot-left), B4 (2x2) (bot-right).
    # 
    # C (3x4) -> C1 (2x2), C2 (2x2), C3 (1x2), C4 (1x2).
    # 
    # C1 = A1*B1 + A2*B3
    # C2 = A1*B2 + A2*B4
    # C3 = A3*B1 + A4*B3
    # C4 = A3*B2 + A4*B4
    # 
    # We can use Strassen for the 2x2x2 multiplications in C1 and C2.
    # C1: 2x2 * 2x2. Rank 7.
    # C2: 2x2 * 2x2. Rank 7.
    # C3: 1x2 * 2x2. This is a vector-matrix product. Rank 4 (schoolbook).
    # C4: 1x2 * 2x2. Rank 4 (schoolbook).
    # Total Rank = 7 + 7 + 4 + 4 = 22? 
    # Wait, the blocks are not independent in the decomposition because the U and V matrices are shared?
    # No, the U and V matrices are for the whole tensor.
    # The decomposition is a sum of rank-1 tensors.
    # If we decompose each block independently, we sum the ranks.
    # So Rank(C1) + Rank(C2) + Rank(C3) + Rank(C4).
    # 
    # Let's verify the indices for this block decomposition.
    # A indices: i (0..2), k (0..3).
    # B indices: k (0..3), j (0..3).
    # C indices: i (0..2), j (0..3).
    # 
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # Block 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    # Block 6: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    # Block 7: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    # Block 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    # 
    # Blocks 1 and 2 contribute to C[0:2, 0:2].
    # Blocks 3 and 4 contribute to C[0:2, 2:4].
    # Blocks 5 and 6 contribute to C[2:3, 0:2].
    # Blocks 7 and 8 contribute to C[2:3, 2:4].
    # 
    # We can decompose each block independently.
    # Block 1: 2x2 * 2x2. Use Strassen (Rank 7).
    # Block 2: 2x2 * 2x2. Use Strassen (Rank 7).
    # Block 3: 2x2 * 2x2. Use Strassen (Rank 7).
    # Block 4: 2x2 * 2x2. Use Strassen (Rank 7).
    # Block 5: 1x2 * 2x2. Schoolbook (Rank 4).
    # Block 6: 1x2 * 2x2. Schoolbook (Rank 4).
    # Block 7: 1x2 * 2x2. Schoolbook (Rank 4).
    # Block 8: 1x2 * 2x2. Schoolbook (Rank 4).
    # 
    # Total Rank = 7*4 + 4*4 = 28 + 16 = 44.
    # 
    # This is a valid rank 44 decomposition!
    # Let's implement this.
    
    def strassen_2x2x2():
        # Returns U, V, W for 2x2 * 2x2 -> 2x2
        # Local indices: i,k in 0..1 for A. k,j in 0..1 for B. i,j in 0..1 for C.
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11) * (B00 + B11) -> C00 += M1, C11 -= M1
        # M2 = (A10 + A11) * B00 -> C10 += M2
        # M3 = A00 * (B01 - B11) -> C01 += M3
        # M4 = A11 * (B10 - B00) -> C11 += M4
        # M5 = (A00 + A01) * B11 -> C01 += M5
        # M6 = (A10 - A00) * (B00 + B01) -> C10 -= M6
        # M7 = (A01 - A11) * (B10 + B11) -> C00 += M7, C11 -= M7
        
        # U indices: i*2 + k
        # V indices: k*2 + j
        # W indices: i*2 + j
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = -1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1
        
        return U, V, W

    def schoolbook_1x2x2():
        # Returns U, V, W for 1x2 * 2x2 -> 1x2
        # Local indices: i=0, k in 0..1 for A. k,j in 0..1 for B. i=0, j in 0..1 for C.
        # U shape (2, 4), V shape (4, 4), W shape (2, 4)
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # C0 = A0*B0 + A1*B2
        # C1 = A0*B1 + A1*B3
        
        # Term 1: A0 * B0 -> C0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 2: A1 * B2 -> C0
        U[1, 1] = 1
        V[2, 1] = 1
        W[0, 1] = 1
        
        # Term 3: A0 * B1 -> C1
        U[0, 2] = 1
        V[1, 2] = 1
        W[1, 2] = 1
        
        # Term 4: A1 * B3 -> C1
        U[1, 3] = 1
        V[3, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Build the full rank 44 solution
    U_full = np.zeros((n * m, 44))
    V_full = np.zeros((m * p, 44))
    W_full = np.zeros((p * n, 44))
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(i_off, k_off, j_off, col_start):
        U_b, V_b, W_b = strassen_2x2x2()
        for r in range(7):
            for a_loc in range(4):
                i_loc = a_loc // 2
                k_loc = a_loc % 2
                a_glob = (i_off + i_loc) * m + (k_off + k_loc)
                U_full[a_glob, col_start + r] += U_b[a_loc, r]
            for b_loc in range(4):
                k_loc = b_loc // 2
                j_loc = b_loc % 2
                b_glob = (k_off + k_loc) * p + (j_off + j_loc)
                V_full[b_glob, col_start + r] += V_b[b_loc, r]
            for c_loc in range(4):
                i_loc = c_loc // 2
                j_loc = c_loc % 2
                c_glob = (j_off + j_loc) * n + (i_off + i_loc)
                W_full[c_glob, col_start + r] += W_b[c_loc, r]
        return col_start + 7
    
    # Helper to add a schoolbook 1x2x2 block
    def add_schoolbook_block(i_row, k_off, j_off, col_start):
        U_b, V_b, W_b = schoolbook_1x2x2()
        for r in range(4):
            for a_loc in range(2):
                k_loc = a_loc
                a_glob = i_row * m + (k_off + k_loc)
                U_full[a_glob, col_start + r] += U_b[a_loc, r]
            for b_loc in range(4):
                k_loc = b_loc // 2
                j_loc = b_loc % 2
                b_glob = (k_off + k_loc) * p + (j_off + j_loc)
                V_full[b_glob, col_start + r] += V_b[b_loc, r]
            for c_loc in range(2):
                j_loc = c_loc
                c_glob = (j_off + j_loc) * n + i_row
                W_full[c_glob, col_start + r] += W_b[c_loc, r]
        return col_start + 4
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 0, 0, col)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 2, 0, col)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_strassen_block(0, 0, 2, col)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    col = add_strassen_block(0, 2, 2, col)
    
    # Block 5: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    col = add_schoolbook_block(2, 0, 0, col)
    
    # Block 6: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    col = add_schoolbook_block(2, 2, 0, col)
    
    # Block 7: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    col = add_schoolbook_block(2, 0, 2, col)
    
    # Block 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    col = add_schoolbook_block(2, 2, 2, col)
    
    if time.time() - _start_time > _deadline:
        return get_schoolbook()
    
    if verify_solution(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # If rank-44 verification fails, fall back to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
