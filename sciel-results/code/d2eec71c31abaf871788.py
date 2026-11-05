"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import numpy as np
from fractions import Fraction
import time

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 -> 2x2 using Strassen (rank 7).
    
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    n=2, m=2, p=2.
    U has 4 rows (i*2+j for i,j in 0..1)
    V has 4 rows (j*2+k for j,k in 0..1)
    W has 4 rows (k*2+i for k,i in 0..1)
    """
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*2+j, c]: A[i,j]
    # V[j*2+k, c]: B[j,k]
    # W[k*2+i, c]: C[i,k]  (since W[k*n+i, c] with n=2, so W[k*2+i])
    
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Column 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0] (k=0,i=0)
    W[3, 0] = 1  # C[1,1] (k=1,i=1)
    
    # Column 1: M2 = (A10+A11)B00
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0] (k=1,i=0... wait W[k*n+i] = W[k*2+i])
    # C[1,0]: k=0, i=1 -> W[0*2+1] = W[1]
    # Let me recheck: W[k*n+i, c] where output C[i,k]. 
    # So for C[1,0]: i=1, k=0 -> W[0*2+1, c] = W[1, c]
    W[1, 1] = 1  # C[1,0]
    # C[1,0] only? M2 contributes to C[1,0] and C[1,1]?
    # C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
    # So M2 contributes to C[1,0] with +1 and C[1,1] with -1
    W[1, 1] = 1   # C[1,0]: k=0, i=1 -> W[0*2+1] = W[1]
    W[2, 1] = -1  # C[1,1]: k=1, i=1 -> W[1*2+1] = W[3]... 
    
    # Let me redo this more carefully.
    # W[k*n + i, c] corresponds to coefficient for C[i, k]
    # C[i, k] is at row i, column k of the output matrix.
    # W index = k * n + i where n=2.
    # So W[k*2 + i] for C[i,k].
    # C[0,0]: k=0,i=0 -> W[0]
    # C[0,1]: k=1,i=0 -> W[2]
    # C[1,0]: k=0,i=1 -> W[1]
    # C[1,1]: k=1,i=1 -> W[3]
    
    # Let me restart U, V, W.
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (A00+A11)(B00+B11)
    # U: A[0,0] and A[1,1] -> indices 0, 3
    # V: B[0,0] and B[1,1] -> indices 0, 3
    # Contributes to C[0,0] (+1) and C[1,1] (+1)
    # W[0] for C[0,0], W[3] for C[1,1]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    # U: A[1,0] and A[1,1] -> indices 2, 3
    # V: B[0,0] -> index 0
    # Contributes to C[1,0] (+1) and C[1,1] (-1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1   # C[1,0]
    W[3, 1] = -1  # C[1,1]
    
    # M3 = A00(B01-B11)
    # U: A[0,0] -> index 0
    # V: B[0,1] and B[1,1] -> indices 1, 3
    # Contributes to C[0,1] (+1) and C[1,1] (+1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1   # C[0,1]
    W[3, 2] = 1   # C[1,1]
    
    # M4 = A11(B10-B00)
    # U: A[1,1] -> index 3
    # V: B[1,0] and B[0,0] -> indices 2, 0
    # Contributes to C[1,0] (+1) and C[0,0] (+1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1   # C[1,0]
    W[0, 3] = 1   # C[0,0]
    
    # M5 = (A00+A10)B11
    # U: A[0,0] and A[1,0] -> indices 0, 2
    # V: B[1,1] -> index 3
    # Contributes to C[0,1] (+1) and C[0,0] (-1)
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[2, 4] = 1   # C[0,1]
    W[0, 4] = -1  # C[0,0]
    
    # M6 = (A11-A00)(B00+B01)
    # U: A[1,1] and A[0,0] -> indices 3, 0
    # V: B[0,0] and B[0,1] -> indices 0, 1
    # Contributes to C[1,1] (+1)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1   # C[1,1]
    
    # M7 = (A01-A11)(B10+B11)
    # U: A[0,1] and A[1,1] -> indices 1, 3
    # V: B[1,0] and B[1,1] -> indices 2, 3
    # Contributes to C[0,0] (+1)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1   # C[0,0]
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] equals the matmul tensor."""
    # Build the tensor T[i,j,k] which should equal sum over a,b of A[i,a]*B[a,k]
    # T[i,j,k] = delta_{i,k} in the tensor sense? No.
    # The tensor for matmul is T[i,j,k] = 1 if the product A[i,j]*B[j,k] appears.
    # Actually the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k).
    # Wait, the tensor T[i,j,k] represents the coefficient of x_i y_j z_k in the product.
    # For matrix multiplication, (AB)[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor is T[i,j,k] = 1 for all i,j,k.
    
    # The decomposition is: sum_c U[i,c] * V[j,c] * W[k,c] should equal T[i,j,k] = 1.
    
    # Compute the decomposed tensor
    # T_dec[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    T_dec = np.einsum('ic,jc,kc->ijk', U, V, W)
    T_target = np.ones((n, m, p))
    
    max_err = np.max(np.abs(T_dec - T_target))
    return max_err

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x3*3x7."""
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = 3, 3, 7
    
    # Strategy: Block decomposition.
    # Decompose 3x3 * 3x7 into blocks.
    # We can view B as having 7 columns. We can split into 3x3, 3x3, 3x1 blocks.
    # Or 3x2, 3x2, 3x2, 3x1.
    
    # Option 1: Split B into [3x2, 3x2, 3x2, 3x1]
    # For each 3x2 block, we need a decomposition of 3x3 * 3x2.
    # Rank of 3x3*3x2: schoolbook is 18. Can we do better?
    # 3x3*3x2: Think of it as 3 rows of A times 3x2 B.
    # Each row of A (1x3) times B (3x2) gives a 1x2 result.
    # This is a 3x3*3x2 tensor.
    
    # Let me try to find a good decomposition for 3x3 * 3x2.
    # A is 3x3, B is 3x2, C is 3x2.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Split A into [A_top(2x3), A_bot(1x3)] and B into [B_top(2x2), B_bot(1x2)].
    # C_top(2x2) = A_top * B_top + A_top * B_bot... no.
    # C[i,k] for i<2: sum_j A[i,j] B[j,k] = sum_{j<2} A[i,j]B[j,k] + A[i,2]B[2,k]
    # C[2,k]: sum_{j<2} A[2,j]B[j,k] + A[2,2]B[2,k]
    
    # Let's use a block approach for 3x3*3x2:
    # Split A into A_1 (2x2 top-left), a_12 (2x1 top-right), a_21 (1x2 bottom-left), a_22 (1x1 bottom-right)
    # Split B into B_1 (2x2 top-left), b_12 (2x1 top-right), b_21 (1x2 bottom-left), b_22 (1x1 bottom-right)
    # But B is 3x2, so:
    # B = [[B11, B12], [B21, B22], [B31, B32]]
    # Let's use 2+1 split on rows.
    # A = [[A11 A12 A13], [A21 A22 A23], [A31 A32 A33]]
    # B = [[B11 B12], [B21 B22], [B31 B32]]
    
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
    # C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k] + A[2,2]B[2,k]
    
    # This is just 18 terms in schoolbook. To reduce, we can use structure.
    # For the top 2x2 part of C (rows 0,1 and both columns):
    # C_top = A_top2 * B_top2 + A_top2 * B_bot + A_bot1 * B_top2... 
    # Actually: C[0:2, :] = A[0:2,0:2] * B[0:2,:] + A[0:2,2] * B[2,:]
    # The first term is 2x2 * 2x2 (Strassen, rank 7).
    # The second term: A[0:2,2] is 2x1, B[2,:] is 1x2. Outer product, rank 4.
    # Similarly for row 2: C[2,:] = A[2,0:2] * B[0:2,:] + A[2,2] * B[2,:]
    # A[2,0:2] is 1x2, B[0:2,:] is 2x2. Rank 4.
    # A[2,2] * B[2,:]: scalar times 1x2, rank 2.
    # Total: 7 + 4 + 4 + 2 = 17.
    
    # So 3x3 * 3x2 has rank at most 17.
    
    # Now for 3x3 * 3x1: schoolbook is 9. Can we do better?
    # C[i] = sum_j A[i,j] B[j]. This is like 3 independent dot products of length 3.
    # Each dot product of two length-3 vectors has rank 3 (schoolbook).
    # Total rank 9. I don't think we can do better for 3x3*3x1 since each output
    # depends on all 3 inputs from both A and B in a separable way.
    # Actually, for a single dot product of length n, the rank is n.
    # So 3 independent dot products of length 3 -> rank 9.
    
    # Total rank for 3x3*3x7 with [3x2, 3x2, 3x2, 3x1] split:
    # 3 * 17 + 9 = 51 + 9 = 60.
    
    # Alternatively, [3x3, 3x3, 3x1] split:
    # 3x3*3x3: rank 21 (Strassen for 3x3? No, 3x3*3x3 rank is 21 or less?)
    # Actually 3x3*3x3 can be done with rank 21? No, schoolbook is 27.
    # There's a rank 21 decomposition for 3x3*3x3? I'm not sure.
    # Let's stick with the [3x2, 3x2, 3x2, 3x1] approach giving rank 60.
    
    # Actually, let me reconsider. Can we do 3x3*3x3 with rank < 27?
    # Known: 2x2*2x2 is rank 7. 3x3*3x3 is at most 21? 
    # I recall that the rank of 3x3*3x3 is between 19 and 23.
    # Let's not rely on that. Use 3x2 blocks.
    
    # Let me implement the 3x3 * 3x2 rank-17 decomposition.
    
    def _matmul_3x3_3x2():
        """Return U, V, W for 3x3 * 3x2 -> 3x2, rank 17.
        
        U: 9 x 17, V: 6 x 17, W: 6 x 17
        """
        n, m, p = 3, 3, 2
        rank = 17
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        
        # Part 1: C[0:2, :] top 2 rows.
        # C[0:2, 0:2] from A[0:2,0:2] * B[0:2,0:2]: Strassen 2x2*2x2, 7 terms.
        U1, V1, W1 = _strassen_2x2()
        # U1 is 4x7, V1 is 4x7, W1 is 4x7
        # Map to global indices:
        # U: A[i,j] for i in 0,1; j in 0,1 -> global U[i*3+j]
        # V: B[j,k] for j in 0,1; k in 0,1 -> global V[j*2+k]
        # W: C[i,k] for i in 0,1; k in 0,1 -> global W[k*3+i] (since W[k*n+i] with n=3)
        
        for c in range(7):
            for local_i, local_j in [(0,0), (1,0), (0,1), (1,1)]:
                if U1[local_i*2+local_j, c] != 0:
                    U[local_i*3 + local_j, col] += U1[local_i*2+local_j, c]
            for local_j, local_k in [(0,0), (1,0), (0,1), (1,1)]:
                if V1[local_j*2+local_k, c] != 0:
                    V[local_j*2 + local_k, col] += V1[local_j*2+local_k, c]
            for local_k, local_i in [(0,0), (1,0), (0,1), (1,1)]:
                if W1[local_k*2+local_i, c] != 0:
                    W[local_k*3 + local_i, col] += W1[local_k*2+local_i, c]
            col += 1
        
        # Part 2: C[0:2, :] from A[0:2,2] * B[2,0:2]: outer product, 4 terms.
        # A[0:2,2] is 2x1: elements A[0,2], A[1,2] -> U[0*3+2]=U[2], U[1*3+2]=U[5]
        # B[2,0:2] is 1x2: elements B[2,0], B[2,1] -> V[2*2+0]=V[4], V[2*2+1]=V[5]
        # C[0:2, 0:2] is 2x2: C[0,0], C[0,1], C[1,0], C[1,1]
        # C[i,k] for i in 0,1; k in 0,1 -> W[k*3+i]
        # C[0,0]: W[0], C[0,1]: W[1], C[1,0]: W[3], C[1,1]: W[4]
        
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] = 1  # A[i,2]
                V[2*2 + k, col] = 1  # B[2,k]
                W[k*3 + i, col] = 1  # C[i,k]
                col += 1
        
        # Part 3: C[2, 0:2] from A[2,0:2] * B[0:2, 0:2]: 1x2 * 2x2, 4 terms.
        # A[2,0:2] is 1x2: A[2,0], A[2,1] -> U[2*3+0]=U[6], U[2*3+1]=U[7]
        # B[0:2, 0:2] is 2x2: B[0,0], B[0,1], B[1,0], B[1,1] -> V[0], V[1], V[2], V[3]
        # C[2, 0:2] is 1x2: C[2,0], C[2,1]
        # C[2,k] for k in 0,1 -> W[k*3+2]
        # C[2,0]: W[2], C[2,1]: W[5]
        
        # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
        for k in range(2):
            for j in range(2):
                U[2*3 + j, col] = 1  # A[2,j]
                V[j*2 + k, col] = 1  # B[j,k]
                W[k*3 + 2, col] = 1  # C[2,k]
                col += 1
        
        # Part 4: C[2, 0:2] from A[2,2] * B[2, 0:2]: scalar * 1x2, 2 terms.
        # A[2,2] -> U[2*3+2]=U[8]
        # B[2,k] -> V[2*2+k]
        # C[2,k] -> W[k*3+2]
        
        for k in range(2):
            U[2*3 + 2, col] = 1  # A[2,2]
            V[2*2 + k, col] = 1  # B[2,k]
            W[k*3 + 2, col] = 1  # C[2,k]
            col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        return U, V, W
    
    def _matmul_3x3_3x1():
        """Return U, V, W for 3x3 * 3x1 -> 3x1, rank 9 (schoolbook).
        
        U: 9 x 9, V: 3 x 9, W: 3 x 9
        """
        n, m, p = 3, 3, 1
        rank = 9
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(1):
                    U[i*3 + j, col] = 1
                    V[j*1 + k, col] = 1
                    W[k*3 + i, col] = 1
                    col += 1
        
        assert col == rank
        return U, V, W
    
    # Now build the full 3x3 * 3x7 decomposition.
    # B has 7 columns. Split into [3x2, 3x2, 3x2, 3x1].
    # Column offsets for B: 0-1, 2-3, 4-5, 6.
    
    total_rank = 3 * 17 + 9  # 60
    
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((21, total_rank))  # m*p = 3*7 = 21
    W_full = np.zeros((21, total_rank))  # p*n = 7*3 = 21
    
    # Get the sub-decompositions
    U_3x2, V_3x2, W_3x2 = _matmul_3x3_3x2()
    U_3x1, V_3x1, W_3x1 = _matmul_3x3_3x1()
    
    col_offset = 0
    
    # Three 3x2 blocks
    for block in range(3):
        b_col_start = block * 2  # Starting column in B for this block
        
        # U is the same for each block (A doesn't change)
        U_full[:, col_offset:col_offset+17] = U_3x2
        
        # V: B[j, k] where j in 0..2, k in b_col_start..b_col_start+1
        # V index = j*7 + k
        for j in range(3):
            for k in range(2):
                local_k = k  # 0 or 1
                V_full[j*7 + b_col_start + k, col_offset:col_offset+17] = V_3x2[j*2 + local_k, :]
        
        # W: C[i, k] where i in 0..2, k in b_col_start..b_col_start+1
        # W index = k*3 + i
        for i in range(3):
            for k in range(2):
                local_k = k  # 0 or 1
                W_full[(b_col_start + k)*3 + i, col_offset:col_offset+17] = W_3x2[k*3 + i, :]
        
        col_offset += 17
    
    # One 3x1 block (column 6)
    U_full[:, col_offset:col_offset+9] = U_3x1
    
    # V: B[j, 6] where j in 0..2
    # V index = j*7 + 6
    for j in range(3):
        V_full[j*7 + 6, col_offset:col_offset+9] = V_3x1[j*1 + 0, :]
    
    # W: C[i, 6] where i in 0..2
    # W index = 6*3 + i
    for i in range(3):
        W_full[6*3 + i, col_offset:col_offset+9] = W_3x1[0*3 + i, :]
    
    col_offset += 9
    assert col_offset == total_rank
    
    # Verify
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        err = _verify_decomposition(3, 3, 7, U_full, V_full, W_full)
        if err > 1e-9:
            # Fallback to schoolbook if verification fails
            rank = 3 * 3 * 7
            U = [[0] * rank for _ in range(9)]
            V = [[0] * rank for _ in range(21)]
            W = [[0] * rank for _ in range(21)]
            column = 0
            for i in range(3):
                for j in range(3):
                    for k in range(7):
                        U[i*3 + j][column] = 1
                        V[j*7 + k][column] = 1
                        W[k*3 + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
