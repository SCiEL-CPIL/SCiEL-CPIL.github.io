<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Use 2x2 Strassen on A blocks, with B split into column groups
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B (4x6) split into B_left (4x4) and B_right (4x2)
    # B_left = [[B11, B12], [B21, B22]] (each 2x2)
    # B_right = [[B13, B14], [B23, B24]] (each 2x2)
    
    # For C_left (4x4), use Strassen: 7 terms
    # For C_right (4x2), use 4 terms (schoolbook on 2x2 blocks)
    # Total: 7 + 4 = 11 terms? But each term is a 2x2 x 2x2 multiplication
    # Wait, no. Each Strassen term is one 2x2x2 multiplication (rank 7 each)
    # So 7 terms * rank 7 = 49 for C_left
    # For C_right: 4 terms * rank 4 (schoolbook 2x2x2) = 16
    # Total: 49 + 16 = 65
    
    # That's better than 96 but not 40.
    
    # Let me implement this:
    # 7 Strassen terms for 4x4x4 (each rank 7) = 49
    # 4 schoolbook terms for 4x4x2 (each rank 4) = 16
    # Total rank = 65
    
    # Actually, for the 4x4x2 part, I can use Strassen too if I pad to 4x4x4
    # But that's wasteful.
    
    # Let me just implement rank 65 and see if it works.
    
    # Hmm, this is getting complicated. Let me just implement a clean Strassen-based decomposition.
    
    # I'll implement the 2x2 Strassen for the 4x4x4 part (rank 49) 
    # and schoolbook for the 4x4x2 part (rank 16)
    # Total rank = 49 + 16 = 65
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 24 x 65
    W = [[0] * rank for _ in range(p * n)]  # 24 x 65
    
    col = 0
    
    # Part 1: 4x4x4 via Strassen (rank 49)
    # Strassen terms:
    # P1 = A11 * B11
    # P2 = A12 * (B21 - B11)
    # P3 = (A11 + A12) * B12
    # P4 = A21 * (B11 - B21)
    # P5 = (A21 + A22) * B22
    # P6 = (A11 - A21) * (B12 + B22)
    # P7 = (A12 - A22) * (B21 - B22)
    
    # Each Pi is a 2x2x2 tensor with rank 7
    # I need to expand each into 7 rank-1 terms
    
    # For a 2x2x2 tensor with A, B, C blocks:
    # C11 = P1 + P7 - P4 + P6
    # C12 = P2 + P4
    # C21 = P3 + P5
    # C22 = P1 - P3 + P2 + P7
    
    # The 7 rank-1 terms for Strassen 2x2x2:
    # s1: A11*B11, contributes to C11 with +1
    # s2: A12*(B21-B11), contributes to C12 with +1
    # s3: (A11+A12)*B12, contributes to C21 with +1
    # s4: A21*(B11-B21), contributes to C12 with +1, C11 with -1
    # s5: (A21+A22)*B22, contributes to C21 with +1
    # s6: (A11-A21)*(B12+B22), contributes to C11 with +1, C22 with +1
    # s7: (A12-A22)*(B21-B22), contributes to C11 with +1, C22 with +1
    
    # Wait, I need to be more careful. Let me define the Strassen decomposition properly.
    
    # For 2x2x2: C = A*B
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]], C = [[c11, c12], [c21, c22]]
    # 
    # M1 = a11 * b11
    # M2 = a12 * (b21 - b11)
    # M3 = (a11 + a12) * b12
    # M4 = a21 * (b11 - b21)
    # M5 = (a21 + a22) * b22
    # M6 = (a11 - a21) * (b12 + b22)
    # M7 = (a12 - a22) * (b21 - b22)
    #
    # c11 = M1 + M7 - M4 + M6
    # c12 = M2 + M4
    # c21 = M3 + M5
    # c22 = M1 - M3 + M2 + M7
    
    # Each M_i is a scalar product, but in the tensor decomposition, each M_i corresponds
    # to a 2x2x2 sub-tensor which itself has rank 7.
    
    # This is getting very complex. Let me just implement schoolbook for now and return rank 96.
    # The measured improvement to rank 40 requires more careful design.
    
    # For this iteration, I'll implement a rank-65 decomposition:
    # 49 for 4x4x4 (Strassen) + 16 for 4x4x2 (schoolbook on 2x2 blocks)
    
    # Actually, let me just return schoolbook rank 96 as a safe baseline.
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
=======
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2_rank1_terms():
    """
    Return the 7 rank-1 terms for the 2x2x2 Strassen decomposition.
    Each term is (U_vec, V_vec, W_vec) where:
      U_vec has 4 entries (indices 0..3 for A11,A12,A21,A22)
      V_vec has 4 entries (indices 0..3 for B11,B12,B21,B22)
      W_vec has 4 entries (indices 0..3 for C11,C12,C21,C22)
    
    Strassen:
      M1 = A11 * B11
      M2 = A12 * (B21 - B11)
      M3 = (A11 + A12) * B12
      M4 = A21 * (B11 - B21)
      M5 = (A21 + A22) * B22
      M6 = (A11 - A21) * (B12 + B22)
      M7 = (A12 - A22) * (B21 - B22)
    
    C11 = M1 + M7 - M4 + M6
    C12 = M2 + M4
    C21 = M3 + M5
    C22 = M1 - M3 + M2 + M7
    """
    # Index: 0=A11/B11/C11, 1=A12/B12/C12, 2=A21/B21/C21, 3=A22/B22/C22
    terms = []
    
    # M1 = A11 * B11
    # U: A11=1, others 0 -> [1,0,0,0]
    # V: B11=1, others 0 -> [1,0,0,0]
    # W: C11+=1, C22+=1 -> [1,0,0,1]
    terms.append(([1,0,0,0], [1,0,0,0], [1,0,0,1]))
    
    # M2 = A12 * (B21 - B11)
    # U: A12=1 -> [0,1,0,0]
    # V: B21=1, B11=-1 -> [-1,0,1,0]
    # W: C12+=1, C22+=1 -> [0,1,0,1]
    terms.append(([0,1,0,0], [-1,0,1,0], [0,1,0,1]))
    
    # M3 = (A11 + A12) * B12
    # U: A11=1, A12=1 -> [1,1,0,0]
    # V: B12=1 -> [0,1,0,0]
    # W: C21+=1, C22-=1 -> [0,0,1,-1]
    terms.append(([1,1,0,0], [0,1,0,0], [0,0,1,-1]))
    
    # M4 = A21 * (B11 - B21)
    # U: A21=1 -> [0,0,1,0]
    # V: B11=1, B21=-1 -> [1,0,-1,0]
    # W: C12+=1, C11-=1 -> [-1,1,0,0]
    terms.append(([0,0,1,0], [1,0,-1,0], [-1,1,0,0]))
    
    # M5 = (A21 + A22) * B22
    # U: A21=1, A22=1 -> [0,0,1,1]
    # V: B22=1 -> [0,0,0,1]
    # W: C21+=1 -> [0,0,1,0]
    terms.append(([0,0,1,1], [0,0,0,1], [0,0,1,0]))
    
    # M6 = (A11 - A21) * (B12 + B22)
    # U: A11=1, A21=-1 -> [1,0,-1,0]
    # V: B12=1, B22=1 -> [0,1,0,1]
    # W: C11+=1, C22+=1 -> [1,0,0,1]
    terms.append(([1,0,-1,0], [0,1,0,1], [1,0,0,1]))
    
    # M7 = (A12 - A22) * (B21 - B22)
    # U: A12=1, A22=-1 -> [0,1,0,-1]
    # V: B21=1, B22=-1 -> [0,0,1,-1]
    # W: C11+=1, C22+=1 -> [1,0,0,1]
    terms.append(([0,1,0,-1], [0,0,1,-1], [1,0,0,1]))
    
    return terms

def _verify_strassen_2x2x2():
    """Verify the Strassen decomposition is correct."""
    terms = _strassen_2x2x2_rank1_terms()
    # Build the tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # For 2x2x2: T should equal the matrix multiplication tensor
    # T[i,j,k] = delta_{i,j} in the sense that T[i,j,k] = 1 if this is a valid MM term
    # Actually, the matrix multiplication tensor M[i,j,k] = delta_{i,k} * ... no
    # M[i,j,k] = 1 if i,j,k form a valid (row, col, intermediate) triple
    # For 2x2x2: M[i,j,k] = 1 for all i,j,k? No.
    # The tensor is: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = delta_{i,?} ... 
    # T[i,j,k] = 1 if j is the intermediate index, i is row of A, k is col of B
    # T[i,j,k] = 1 for all valid (i,j,k) where i in [0,1], j in [0,1], k in [0,1]
    # Wait no. The tensor T satisfies: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # For standard MM: T[i,j,k] = delta_{j,?}... 
    # Actually T[i,j,k] = 1 if the indices match: row i of A times column k of B through intermediate j
    # T[i,j,k] = 1 for all i,j,k in {0,1} since every (i,j,k) is a valid term in 2x2 MM
    # No wait: C[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all i,j,k.
    # That's not right either. Let me think again.
    # The tensor M has entries M[i,j,k] where the decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: M = sum_{i,j,k} M[i,j,k] e_i ⊗ e_j ⊗ e_k
    # where the coefficient of e_i ⊗ e_j ⊗ e_k in M is 1 if (i,j,k) is a valid triple.
    # For matrix multiplication, M[i,j,k] = 1 for all i,j,k (since every combination contributes).
    # Wait, that gives rank >= 8 which is wrong. Let me reconsider.
    # 
    # The matrix multiplication tensor is: M[i,j,k] = delta_{i,?}...
    # Actually, the standard formulation: the tensor T satisfies
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k] (for each i,k)
    # So T[i,j,k] = 1 for all i,j,k. But that's an 8-dimensional tensor of all ones,
    # which has rank 8 (schoolbook). But Strassen says rank 7!
    #
    # I think I'm confusing myself. Let me be very explicit.
    # The tensor M ∈ R^{2×2×2} is defined by M[i,j,k] = 1 for all i,j,k ∈ {0,1}.
    # No wait, that's not right. The tensor for 2x2 matrix multiplication is:
    # M[i,j,k] = 1 if and only if the indices form a valid multiplication term.
    # For C = AB where C[i,k] = sum_j A[i,j]B[j,k]:
    # The tensor T has T[i,j,k] = 1 for all i,j,k.
    # Hmm, but then the rank-1 decomposition would be:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r[i]*v_r[j]*w_r[k] sums to T[i,j,k] = 1 for all i,j,k.
    # 
    # Actually I think the tensor is: T[i,j,k] = delta_{i,k}... no.
    # Let me just check: for 2x2 MM, the tensor is the 2x2x2 all-ones tensor? No.
    # 
    # OK let me think differently. The tensor M is such that:
    # <M, A ⊗ B> = C where C[i,k] = sum_j A[i,j] B[j,k]
    # So M[i,j,k] * A[i,j] * B[j,k] summed over i,j,k gives... 
    # sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * something?
    # 
    # I think the correct formulation is:
    # The tensor M ∈ R^{I×J×K} where I=rows of A, J=cols of A=rows of B, K=cols of B
    # M[i,j,k] = 1 for all i,j,k.
    # And the decomposition M = sum_r u_r ⊗ v_r ⊗ w_r means:
    # M[i,j,k] = sum_r u_r[i] v_r[j] w_r[k]
    # For this to equal 1 for all i,j,k, we need the rank-1 vectors to span appropriately.
    #
    # Wait, I think I have it backwards. Let me look at this more carefully.
    # 
    # The matrix multiplication tensor T ∈ R^{n×m×p} is defined by:
    # T[i,j,k] = 1 if (i,j,k) is a valid index triple, 0 otherwise.
    # For n×m times m×p: T[i,j,k] = 1 for all i∈[n], j∈[m], k∈[p].
    # So it's the all-ones tensor!
    # 
    # No, that can't be right because then the rank would be 1 (just u=v=w=ones vector).
    # 
    # I think the issue is in how the tensor is defined. Let me use the standard definition:
    # The tensor T satisfies: for all A (n×m), B (m×p):
    #   C = A*B means C[i,k] = sum_j A[i,j] B[j,k]
    #   This is equivalent to: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * (something)
    #
    # Actually, I think the correct tensor definition is:
    # T[i,j,k] = delta_{i,?}... 
    #
    # Let me just directly verify my Strassen terms by checking that they reproduce MM.
    # For 2x2: C = A*B
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # Using Strassen:
    # M1 = A[0,0]*B[0,0]
    # M2 = A[0,1]*(B[1,0]-B[0,0])
    # M3 = (A[0,0]+A[0,1])*B[0,1]
    # M4 = A[1,0]*(B[0,0]-B[1,0])
    # M5 = (A[1,0]+A[1,1])*B[1,1]
    # M6 = (A[0,0]-A[1,0])*(B[0,1]+B[1,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]-B[1,1])
    #
    # C[0,0] = M1 + M7 - M4 + M6
    # = A[0,0]B[0,0] + (A[0,1]-A[1,1])(B[1,0]-B[1,1]) - A[1,0](B[0,0]-B[1,0]) + (A[0,0]-A[1,0])(B[0,1]+B[1,1])
    # = A[0,0]B[0,0] + A[0,1]B[1,0] - A[0,1]B[1,1] - A[1,1]B[1,0] + A[1,1]B[1,1] - A[1,0]B[0,0] + A[1,0]B[1,0] + A[0,0]B[0,1] + A[0,0]B[1,1] - A[1,0]B[0,1] - A[1,0]B[1,1]
    # 
    # This should equal A[0,0]B[0,0] + A[0,1]B[1,0]. Let me check if the extra terms cancel...
    # This is getting messy. Let me just trust the standard Strassen formulas and verify numerically.
    pass

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Block decomposition:
    # A is 4x4, split into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # B is 4x6, split into 2x2 blocks: B = [[B11, B12, B13], [B21, B22, B23]]
    #   where B11, B12, B13 are 2x2, 2x2, 2x2
    # C is 4x6, split into 2x2 blocks: C = [[C11, C12, C13], [C21, C22, C23]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    #
    # Strategy: Use Strassen for the 4x4x4 part (columns 0-3 of B) and schoolbook for 4x4x2 (columns 4-5)
    #
    # For the 4x4x4 part:
    #   C_left = [[C11, C12], [C21, C22]] = A * B_left where B_left = [[B11, B12], [B21, B22]]
    #   Use Strassen: 7 block multiplications, each 2x2x2 with rank 7, total 49
    #
    # For the 4x4x2 part:
    #   C_right = [[C13, C13'], [C23, C23']] where B_right = [[B13, B13'], [B23, B23']]
    #   Wait, B is 4x6 so B_right is 4x2. Let me re-index.
    #   B columns 0,1 -> B11 (rows 0,1) and B21 (rows 2,3)
    #   B columns 2,3 -> B12 (rows 0,1) and B22 (rows 2,3)
    #   B columns 4,5 -> B13 (rows 0,1) and B23 (rows 2,3)
    #
    #   C13 = A11*B13 + A12*B23 (2x2 = 2x2*2x2 + 2x2*2x2)
    #   C23 = A21*B13 + A22*B23
    #   Schoolbook: 4 terms of 2x2x2, each rank 4, total 16
    #
    # Total rank: 49 + 16 = 65
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 24 x 65
    W = [[0] * rank for _ in range(p * n)]  # 24 x 65
    
    # Helper: add a rank-1 term
    def add_term(U_vec, V_vec, W_vec, col):
        for i in range(len(U_vec)):
            if U_vec[i] != 0:
                U[i][col] += U_vec[i]
        for j in range(len(V_vec)):
            if V_vec[j] != 0:
                V[j][col] += V_vec[j]
        for k in range(len(W_vec)):
            if W_vec[k] != 0:
                W[k][col] += W_vec[k]
    
    # Index mapping for 4x4 matrix A:
    # A[row][col] -> U index = row * 4 + col
    # Block A11: rows 0-1, cols 0-1 -> U indices 0,1,4,5
    # Block A12: rows 0-1, cols 2-3 -> U indices 2,3,6,7
    # Block A21: rows 2-3, cols 0-1 -> U indices 8,9,12,13
    # Block A22: rows 2-3, cols 2-3 -> U indices 10,11,14,15
    
    # Index mapping for 4x6 matrix B:
    # B[row][col] -> V index = row * 6 + col
    # Block B11: rows 0-1, cols 0-1 -> V indices 0,1,6,7
    # Block B12: rows 0-1, cols 2-3 -> V indices 2,3,8,9
    # Block B13: rows 0-1, cols 4-5 -> V indices 4,5,10,11
    # Block B21: rows 2-3, cols 0-1 -> V indices 12,13,18,19
    # Block B22: rows 2-3, cols 2-3 -> V indices 14,15,20,21
    # Block B23: rows 2-3, cols 4-5 -> V indices 16,17,22,23
    
    # Index mapping for 4x6 matrix C:
    # C[row][col] -> W index = col * 4 + row (trace convention: W[k*n+i, r])
    # Block C11: rows 0-1, cols 0-1 -> W indices 0,1,4,5
    # Block C12: rows 0-1, cols 2-3 -> W indices 2,3,6,7
    # Block C13: rows 0-1, cols 4-5 -> W indices 8,9,12,13
    # Block C21: rows 2-3, cols 0-1 -> W indices 14,15,18,19
    # Block C22: rows 2-3, cols 2-3 -> W indices 16,17,20,21
    # Block C23: rows 2-3, cols 4-5 -> W indices 22,23,26,27
    
    # Wait, W has shape (p*n, rank) = (24, rank). W index = k * n + i = k * 4 + i
    # C[i,k] where i is row (0-3), k is col (0-5)
    # W index for C[i,k] = k * 4 + i
    
    # Block C11: i in {0,1}, k in {0,1} -> W indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # Block C12: i in {0,1}, k in {2,3} -> W indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # Block C13: i in {0,1}, k in {4,5} -> W indices: 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21
    # Block C21: i in {2,3}, k in {0,1} -> W indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # Block C22: i in {2,3}, k in {2,3} -> W indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    # Block C23: i in {2,3}, k in {4,5} -> W indices: 4*4+2=18, 4*4+3=19, 5*4+2=22, 5*4+3=23
    
    # Let me define index maps for each block
    # For a 2x2 block at (row_block, col_block) in a matrix with row_dim and col_dim:
    # element (r,c) within block -> global index
    
    # A block indices (row_block, col_block) -> [idx00, idx01, idx10, idx11]
    A_blocks = {
        (0,0): [0, 1, 4, 5],    # A11
        (0,1): [2, 3, 6, 7],    # A12
        (1,0): [8, 9, 12, 13],  # A21
        (1,1): [10, 11, 14, 15] # A22
    }
    
    # B block indices (row_block, col_block) -> [idx00, idx01, idx10, idx11]
    B_blocks = {
        (0,0): [0, 1, 6, 7],    # B11
        (0,1): [2, 3, 8, 9],    # B12
        (0,2): [4, 5, 10, 11],  # B13
        (1,0): [12, 13, 18, 19],# B21
        (1,1): [14, 15, 20, 21],# B22
        (1,2): [16, 17, 22, 23] # B23
    }
    
    # C block indices (row_block, col_block) -> [idx00, idx01, idx10, idx11]
    # C[i,k] -> W index k*4+i
    C_blocks = {
        (0,0): [0, 1, 4, 5],    # C11: i=0,1; k=0,1
        (0,1): [8, 9, 12, 13],  # C12: i=0,1; k=2,3
        (0,2): [16, 17, 20, 21],# C13: i=0,1; k=4,5
        (1,0): [2, 3, 6, 7],    # C21: i=2,3; k=0,1
        (1,1): [10, 11, 14, 15],# C22: i=2,3; k=2,3
        (1,2): [18, 19, 22, 23] # C23: i=2,3; k=4,5
    }
    
    strassen_terms = _strassen_2x2x2_rank1_terms()
    
    col = 0
    
    # Part 1: 4x4x4 via Strassen (rank 49)
    # Strassen computes C_left = A * B_left
    # where B_left = [[B11, B12], [B21, B22]]
    # 
    # The 7 Strassen block multiplications:
    # M1 = A11 * B11
    # M2 = A12 * (B21 - B11)
    # M3 = (A11 + A12) * B12
    # M4 = A21 * (B11 - B21)
    # M5 = (A21 + A22) * B22
    # M6 = (A11 - A21) * (B12 + B22)
    # M7 = (A12 - A22) * (B21 - B22)
    #
    # C11 = M1 + M7 - M4 + M6
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M3 + M2 + M7
    #
    # Each Mi is a 2x2 matrix product, decomposed into 7 rank-1 terms.
    # For each Strassen term, we need to:
    # 1. Determine which A block(s) and B block(s) are involved
    # 2. For each of the 7 rank-1 sub-terms, create a global rank-1 term
    
    # For Strassen term with A-factor and B-factor:
    # The A-factor is a linear combination of A-blocks
    # The B-factor is a linear combination of B-blocks
    # The result contributes to C-blocks with certain coefficients
    
    # Let me define the 7 Strassen terms as (A_factor, B_factor, C_contribution)
    # A_factor: dict from block_index to coefficient
    # B_factor: dict from block_index to coefficient  
    # C_contribution: dict from block_index to coefficient
    
    # M1 = A11 * B11
    #   A: {(0,0): 1}
    #   B: {(0,0): 1}
    #   C: contributes to C11 with +1, C22 with +1
    #   C: {(0,0): +1, (1,1): +1}
    
    # M2 = A12 * (B21 - B11)
    #   A: {(0,1): 1}
    #   B: {(1,0): 1, (0,0): -1}
    #   C: contributes to C12 with +1, C22 with +1
    #   C: {(0,1): +1, (1,1): +1}
    
    # M3 = (A11 + A12) * B12
    #   A: {(0,0): 1, (0,1): 1}
    #   B: {(0,1): 1}
    #   C: contributes to C21 with +1, C22 with -1
    #   C: {(1,0): +1, (1,1): -1}
    
    # M4 = A21 * (B11 - B21)
    #   A: {(1,0): 1}
    #   B: {(0,0): 1, (1,0): -1}
    #   C: contributes to C12 with +1, C11 with -1
    #   C: {(0,1): +1, (0,0): -1}
    
    # M5 = (A21 + A22) * B22
    #   A: {(1,0): 1, (1,1): 1}
    #   B: {(1,1): 1}
    #   C: contributes to C21 with +1
    #   C: {(1,0): +1}
    
    # M6 = (A11 - A21) * (B12 + B22)
    #   A: {(0,0): 1, (1,0): -1}
    #   B: {(0,1): 1, (1,1): 1}
    #   C: contributes to C11 with +1, C22 with +1
    #   C: {(0,0): +1, (1,1): +1}
    
    # M7 = (A12 - A22) * (B21 - B22)
    #   A: {(0,1): 1, (1,1): -1}
    #   B: {(1,0): 1, (1,1): -1}
    #   C: contributes to C11 with +1, C22 with +1
    #   C: {(0,0): +1, (1,1): +1}
    
    strassen_defs = [
        # (A_factor, B_factor, C_contribution)
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1, (1,1): 1}),
        ({(0,1): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(0,1): 1}, {(1,0): 1, (1,1): -1}),
        ({(1,0): 1}, {(0,0): 1, (1,0): -1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (1,1): 1}, {(1,1): 1}, {(1,0): 1}),
        ({(0,0): 1, (1,0): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): -1}, {(0,0): 1, (1,1): 1}),
    ]
    
    for A_factor, B_factor, C_contrib in strassen_defs:
        for u_sub, v_sub, w_sub in strassen_terms:
            # u_sub = [u00, u01, u10, u11] for the A-block
            # v_sub = [v00, v01, v10, v11] for the B-block
            # w_sub = [w00, w01, w10, w11] for the C-block
            #
            # For each element of u_sub, multiply by the corresponding A_factor coefficient
            # and add to the appropriate U index
            U_vec = [0] * (n * m)
            V_vec = [0] * (m * p)
            W_vec = [0] * (p * n)
            
            # u_sub indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) within the block
            # Map to A_factor keys
            sub_to_block = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
            
            for si in range(4):
                if u_sub[si] != 0:
                    # This corresponds to element (si//2, si%2) within the A-block
                    # We need to add u_sub[si] * A_factor[block] to the appropriate U indices
                    # But A_factor maps block->coeff, and the block has 4 elements
                    # The element (r,c) in block (br,bc) is at global index:
                    # For A: row = br*2+r, col = bc*2+c, index = row*4+col
                    for ab_key, ab_coeff in A_factor.items():
                        # ab_key is the block in A
                        # u_sub[si] corresponds to position si in the "local" 2x2
                        # But we need to figure out which global A block this is
                        # Actually, u_sub is for a specific A block structure
                        # Let me reconsider...
                        pass
            
            # This approach is getting too complicated. Let me use a different strategy.
            # For each Strassen term, the A-factor is a linear combination of A-blocks.
            # Each A-block is 2x2, so the A-factor is a 4x4 matrix (with most entries zero).
            # Similarly for B-factor (4x6 with most entries zero) and C-contribution (4x4).
            #
            # Then for each of the 7 rank-1 sub-terms of the 2x2x2 Strassen:
            # The sub-term has u_sub (4 entries for a 2x2), v_sub (4 entries for a 2x2), w_sub (4 entries for a 2x2)
            # We need to "expand" these to the full 4x4, 4x6, 4x4 sizes by placing them in the correct blocks.
            
            # Let me define the full A-factor as a 4x4 matrix
            A_full = [[0]*4 for _ in range(4)]
            for (br, bc), coeff in A_factor.items():
                for r in range(2):
                    for c in range(2):
                        A_full[br*2+r][bc*2+c] = coeff
            
            B_full = [[0]*6 for _ in range(4)]
            for (br, bc), coeff in B_factor.items():
                for r in range(2):
                    for c in range(2):
                    # B is 4x6, block (br,bc) is at rows br*2..br*2+1, cols bc*2..bc*2+1
                        B_full[br*2+r][bc*2+c] = coeff
            
            C_full = [[0]*4 for _ in range(4)]
            for (br, bc), coeff in C_contrib.items():
                for r in range(2):
                    for c in range(2):
                        C_full[br*2+r][bc*2+c] = coeff
            
            # Now u_sub, v_sub, w_sub are 4-element vectors for 2x2 blocks
            # u_sub[i] corresponds to element i in row-major order of the 2x2
            # But which 2x2 block? It's the "local" block within the Strassen structure.
            # 
            # Actually, I think the 7 rank-1 terms for Strassen 2x2x2 are defined on the 2x2x2 tensor.
            # u_sub[i] is the coefficient for A element i (in 2x2), v_sub[j] for B element j, w_sub[k] for C element k.
            #
            # To expand to the full problem, I need to figure out which global indices correspond.
            # The Strassen 2x2x2 operates on blocks, so:
            # u_sub[0] -> A11[0,0], u_sub[1] -> A11[0,1], u_sub[2] -> A11[1,0], u_sub[3] -> A11[1,1]
            # But wait, that's only for one block. The Strassen terms involve different blocks.
            #
            # I think the correct interpretation is:
            # The 7 rank-1 terms are for a generic 2x2x2 tensor.
            # When we apply Strassen to 4x4x4, each "scalar" in the 2x2x2 is actually a 2x2 matrix.
            # So the 7 rank-1 terms of the 2x2x2 become 7 * 7 = 49 rank-1 terms of the 4x4x4.
            #
            # For Strassen term s (one of the 7), with A-factor A_s (4x4), B-factor B_s (4x4), C-factor C_s (4x4):
            # For each of the 7 rank-1 sub-terms (u_sub, v_sub, w_sub):
            #   The full rank-1 term is: U_full = A_s ⊗ u_sub (Kronecker-like), etc.
            #   But it's not exactly Kronecker product. Let me think...
            #
            # Actually, I think the expansion is:
            # U_full[i,j] = A_s[i,j] * u_sub[local_index(i,j)]
            # where local_index maps (i,j) to the position within the 2x2 block.
            # But A_s is sparse (only one or two blocks are nonzero), so this should work.
            #
            # Let me try: for each (i,j) in 4x4:
            #   If A_s[i,j] != 0, then it belongs to some block (br,bc) where i=br*2+r, j=bc*2+c
            #   The local index is r*2+c (row-major in the 2x2 block)
            #   U_full[i,j] = A_s[i,j] * u_sub[r*2+c]
            
            U_full = [0.0] * 16
            for i in range(4):
                for j in range(4):
                    if A_full[i][j] != 0:
                        br = i // 2
                        bc = j // 2
                        r = i % 2
                        c = j % 2
                        local_idx = r * 2 + c
                        U_full[i * 4 + j] = A_full[i][j] * u_sub[local_idx]
            
            V_full = [0.0] * 24
            for i in range(4):
                for j in range(6):
                    if B_full[i][j] != 0:
                        br = i // 2
                        bc = j // 2
                        r = i % 2
                        c = j % 2
                        local_idx = r * 2 + c
                        V_full[i * 6 + j] = B_full[i][j] * v_sub[local_idx]
            
            W_full = [0.0] * 16
            for i in range(4):
                for j in range(4):
                    if C_full[i][j] != 0:
                        br = i // 2
                        bc = j // 2
                        r = i % 2
                        c = j % 2
                        local_idx = r * 2 + c
                        W_full[j * 4 + i] = C_full[i][j] * w_sub[local_idx]
            
            add_term(U_full, V_full, W_full, col)
            col += 1
    
    # Part 2: 4x4x2 via schoolbook (rank 16)
    # C13 = A11*B13 + A12*B23
    # C23 = A21*B13 + A22*B23
    #
    # Schoolbook for 2x2x2: 4 terms
    # For A11*B13: 4 terms
    # For A12*B23: 4 terms
    # For A21*B13: 4 terms
    # For A22*B23: 4 terms
    # Total: 16 terms
    
    schoolbook_2x2x2 = [
        ([1,0,0,0], [1,0,0,0], [1,0,0,0]),  # a00*b00 -> c00
        ([0,1,0,0], [1,0,0,0], [0,1,0,0]),  # a01*b10 -> c01
        ([1,0,0,0], [0,0,1,0], [0,0,1,0]),  # a00*b01 -> c01  wait
        # Let me redo schoolbook 2x2x2:
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        # Rank-1 terms:
        # (a00,b00,c00): U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]
        # (a01,b10,c00): U=[0,1,0,0], V=[0,0,1,0], W=[1,0,0,0]
        # (a00,b01,c01): U=[1,0,0,0], V=[0,1,0,0], W=[0,1,0,0]
        # (a01,b11,c01): U=[0,1,0,0], V=[0,0,0,1], W=[0,1,0,0]
        # (a10,b00,c10): U=[0,0,1,0], V=[1,0,0,0], W=[0,0,1,0]
        # (a11,b10,c10): U=[0,0,0,1], V=[0,0,1,0], W=[0,0,1,0]
        # (a10,b01,c11): U=[0,0,1,0], V=[0,1,0,0], W=[0,0,0,1]
        # (a11,b11,c11): U=[0,0,0,1], V=[0,0,0,1], W=[0,0,0,1]
    ]
    
    # Wait, that's 8 terms for 2x2x2 schoolbook, not 4. The rank of 2x2x2 MM tensor is 7 (Strassen) or 8 (schoolbook).
    # So for the 4x4x2 part, I need:
    # C13 = A11*B13 + A12*B23: each is 2x2x2, rank 7 each -> 14 terms? No...
    # Actually, C13 is a 2x2 matrix = A11(2x2) * B13(2x2) + A12(2x2) * B23(2x2)
    # This is a sum of two 2x2x2 tensors. Each has rank 7 (Strassen) or 8 (schoolbook).
    # But they share the same C block, so I can't just add ranks.
    #
    # Actually, C13 = A11*B13 + A12*B23 is a 2x2 matrix computed from four 2x2 blocks.
    # The tensor for this is in R^{2x2 x 2x2 x 2x2} but with specific structure.
    # Let me think of it as: C13[i,k] = sum_j (A11[i,j]*B13[j,k] + A12[i,j]*B23[j,k])
    # This is a 2x2x2 tensor (i,j,k) with entries:
    # T[i,j,k] = delta_{i,?}... 
    # T[i,j,k] = 1 if we're computing C13[i,k] using A11[i,j]*B13[j,k] or A12[i,j]*B23[j,k]
    # 
    # Actually, let me just use schoolbook directly on the 4x4x2 problem.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {4,5} (for C13)
    # and i in {2,3}, k in {4,5} (for C23)
    #
    # For C13 (2x2 output): 2*2*2 = 8 scalar multiplications (schoolbook)
    # For C23 (2x2 output): 2*2*2 = 8 scalar multiplications
    # Total: 16 rank-1 terms
    
    # C13[i,k] where i in {0,1}, k in {4,5}
    # C13[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4] + A[0,3]*B[3,4]
    # C13[0,5] = A[0,0]*B[0,5] + A[0,1]*B[1,5] + A[0,2]*B[2,5] + A[0,3]*B[3,5]
    # C13[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4] + A[1,2]*B[2,4] + A[1,3]*B[3,4]
    # C13[1,5] = A[1,0]*B[0,5] + A[1,1]*B[1,5] + A[1,2]*B[2,5] + A[1,3]*B[3,5]
    
    # C23[2,4] = A[2,0]*B[0,4] + A[2,1]*B[1,4] + A[2,2]*B[2,4] + A[2,3]*B[3,4]
    # C23[2,5] = A[2,0]*B[0,5] + A[2,1]*B[1,5] + A[2,2]*B[2,5] + A[2,3]*B[3,5]
    # C23[3,4] = A[3,0]*B[0,4] + A[3,1]*B[1,4] + A[3,2]*B[2,4] + A[3,3]*B[3,4]
    # C23[3,5] = A[3,0]*B[0,5] + A[3,1]*B[1,5] + A[3,2]*B[2,5] + A[3,3]*B[3,5]
    
    # Each scalar product A[i,j]*B[j,k] is one rank-1 term.
    # Total: 4 * 4 = 16 terms for the 4x4x2 part.
    
    for i in range(4):
        for j in range(4):
            for k in [4, 5]:
                U_vec = [0] * 16
                V_vec = [0] * 24
                W_vec = [0] * 24
                U_vec[i * 4 + j] = 1
                V_vec[j * 6 + k] = 1
                W_vec[k * 4 + i] = 1
                add_term(U_vec, V_vec, W_vec, col)
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i,r] * V[j,r] * W[k,r] should equal the MM tensor
    # MM tensor T[i,j,k] = 1 for all i in [0,3], j in [0,3], k in [0,5]
    # Actually, the tensor is: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T satisfies: T[i,j,k] = 1 for all valid (i,j,k)
    # In the decomposition: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    
    # Let's verify on a random example
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 6)
    C_test = A_test @ B_test
    
    # Compute C from decomposition
    C_decomp = np.zeros((4, 6))
    for i in range(4):
        for k in range(6):
            total = 0.0
            for j in range(4):
                for r in range(rank):
                    total += U_arr[i*4+j, r] * V_arr[j*6+k, r] * W_arr[k*4+i, r]
            C_decomp[i, k] = total
    
    if not np.allclose(C_decomp, C_test, atol=1e-10):
        # Try to debug
        max_err = np.max(np.abs(C_decomp - C_test))
        raise ValueError(f"Decomposition verification failed! Max error: {max_err}")
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

Wait, I realize my Strassen expansion logic has a flaw. The `u_sub` vector from the 2x2x2 Strassen terms is defined for a generic 2x2 block, but when I have multiple A-blocks in the factor, I need to correctly map each sub-index to the right global position. Let me fix this with a cleaner approach.

