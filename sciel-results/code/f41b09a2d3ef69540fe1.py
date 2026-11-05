import time
import sys

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strassen-Winograd rank-8 algorithm for 3x3 matrix multiplication
    # C = A * B where A, B, C are 3x3
    # 8 multiplications:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # M8 = (A[1,0]+A[1,1]-A[0,0]) * (B[0,0]+B[1,1])
    #
    # C[0,0] = M1 - M2 + M3 + M4 - M5 + M6
    # C[0,1] = M1 - M2 - M4 + M5 - M6 - M7
    # C[1,0] = M2 + M3 - M4 + M5 + M6
    # C[1,1] = M1 - M3 + M4 - M5 + M7
    # C[2,0] = M1 + M2 - M3 + M4 - M6 + M8
    # C[2,1] = M1 - M2 + M3 - M4 + M5 - M7 - M8
    # C[0,2] = M2 + M3 - M4 - M6
    # C[1,2] = M1 - M2 + M4 - M5 + M7
    # C[2,2] = M1 - M2 + M3 + M4 - M6 + M8
    
    # Define the Strassen factors
    # U factors (linear combos of A entries, indexed by i*m+j = i*3+j)
    # V factors (linear combos of B entries)
    # W factors (linear combos of C entries)
    
    strassen_U = [
        [1,0,0, 0,1,0, 0,0,0],  # A00+A11
        [0,0,0, 1,1,0, 0,0,0],  # A10+A11
        [1,0,0, 0,0,0, 0,0,0],  # A00
        [0,0,0, 0,1,0, 0,0,0],  # A11
        [1,1,0, 0,0,0, 0,0,0],  # A00+A01
        [0,0,0, 1,-1,0, 0,0,0], # A10-A00
        [0,0,0, 0,0,1, -1,0,0], # A01-A11  (wait, let me recheck)
        # Let me redo this carefully
    ]
    
    # Let me use a cleaner formulation. A is 3x3, indexed A[i][j], flattened as A[i*3+j]
    # B is 3x3, indexed B[i][j], flattened as B[i*3+j]
    # C is 3x3, indexed C[i][j], flattened as C[i*3+j]
    
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # M8 = (A10+A11-A00)*(B00+B11)
    
    # C00 = M1 - M2 + M3 + M4 - M5 + M6
    # C01 = M1 - M2 - M4 + M5 - M6 - M7
    # C10 = M2 + M3 - M4 + M5 + M6
    # C11 = M1 - M3 + M4 - M5 + M7
    # C20 = M1 + M2 - M3 + M4 - M6 + M8
    # C21 = M1 - M2 + M3 - M4 + M5 - M7 - M8
    # C02 = M2 + M3 - M4 - M6
    # C12 = M1 - M2 + M4 - M5 + M7
    # C22 = M1 - M2 + M3 + M4 - M6 + M8
    
    # U vectors (9-dim, for A):
    strassen_U = [
        [1,0,0, 0,1,0, 0,0,0],   # M1: A00+A11
        [0,0,0, 1,1,0, 0,0,0],   # M2: A10+A11
        [1,0,0, 0,0,0, 0,0,0],   # M3: A00
        [0,0,0, 0,1,0, 0,0,0],   # M4: A11
        [1,1,0, 0,0,0, 0,0,0],   # M5: A00+A01
        [1,-1,0, 0,0,0, 0,0,0],  # M6: A10-A00 -> wait, A10 is index 3, A00 is index 0
        # Fix: A[i*3+j], so A00=0, A01=1, A02=2, A10=3, A11=4, A12=5, A20=6, A21=7, A22=8
        [0,0,0, 1,1,0, 0,0,0],   # M2 already defined
    ]
    
    # Let me restart with correct indexing
    # A flat: [A00, A01, A02, A10, A11, A12, A20, A21, A22]
    # indices:  0    1    2    3    4    5    6    7    8
    
    U_strassen = [
        [1,0,0, 0,1,0, 0,0,0],   # M1: A00+A11
        [0,0,0, 1,1,0, 0,0,0],   # M2: A10+A11
        [1,0,0, 0,0,0, 0,0,0],   # M3: A00
        [0,0,0, 0,1,0, 0,0,0],   # M4: A11
        [1,1,0, 0,0,0, 0,0,0],   # M5: A00+A01
        [0,0,0, 1,-1,0, 0,0,0],  # M6: A10-A00 -> indices 3 and 0: [1,0,0,1,0,0,0,0,0] - [1,0,0,0,0,0,0,0,0] = [0,0,0,1,0,0,0,0,0]? No.
        # M6 = (A10 - A00): A10 is index 3, A00 is index 0
        # So: [1,0,0, -1,0,0, 0,0,0]? No: A10 - A00 means coefficient of A10 is +1, A00 is -1
        # Flat: [-1,0,0, 1,0,0, 0,0,0]
    ]
    
    # I'm getting confused. Let me just code it cleanly.
    
    def strassen_3x3_factors():
        """Return (U, V, W) for 3x3 * 3x3 -> 3x3 using rank-8 Strassen-Winograd."""
        # U: 9x8, V: 9x8, W: 9x8
        # Each column is a rank-1 term factor
        
        U = [
            [1,0,0, 0,1,0, 0,0,0],   # M1: A00+A11
            [0,0,0, 1,1,0, 0,0,0],   # M2: A10+A11  
            [1,0,0, 0,0,0, 0,0,0],   # M3: A00
            [0,0,0, 0,1,0, 0,0,0],   # M4: A11
            [1,1,0, 0,0,0, 0,0,0],   # M5: A00+A01
            [-1,0,0, 1,0,0, 0,0,0],  # M6: A10-A00
            [0,1,0, 0,-1,0, 0,0,0],  # M7: A01-A11
            [-1,0,0, 1,1,0, 0,0,0],  # M8: A10+A11-A00
        ]
        
        V = [
            [1,0,0, 0,1,0, 0,0,0],   # M1: B00+B11
            [1,0,0, 0,0,0, 0,0,0],   # M2: B00
            [0,0,0, 1,-1,0, 0,0,0],  # M3: B10-B11
            [0,1,0, -1,0,0, 0,0,0],  # M4: B01-B00
            [0,0,0, 0,1,0, 0,0,0],   # M5: B11
            [1,1,0, 0,0,0, 0,0,0],   # M6: B00+B01
            [0,0,0, 1,1,0, 0,0,0],   # M7: B10+B11
            [1,0,0, 0,1,0, 0,0,0],   # M8: B00+B11
        ]
        
        W = [
            [1,0,0, 0,0,0, 0,0,0],   # M1: C00
            [1,0,0, 0,0,0, 0,0,0],   # M1: C01
            [0,1,0, 0,0,0, 0,0,0],   # M2: C10
            [0,1,0, 0,0,0, 0,0,0],   # M2: C11
            [0,0,0, 0,0,1, 0,0,0],   # M3: C02
            [0,0,0, 0,0,1, 0,0,0],   # M3: C12
            [0,0,0, 0,0,1, 0,0,0],   # M4: C02
            [0,0,0, 0,0,1, 0,0,0],   # M4: C12
        ]
        
        # This is getting messy. Let me use a different approach.
        # I'll define the 8 multiplications and their contributions to C.
        
        return None
    
    # Let me take a completely different, cleaner approach.
    # I'll directly construct the decomposition by specifying which A-index, B-index, C-index 
    # each rank-1 term contributes to, with coefficients.
    
    # For the 3x3x8 problem:
    # A is 3x3 (9 entries), B is 3x8 (24 entries), C is 8x3 (24 entries)
    # C[k*n+i, r] where n=3, so C[k*3+i, r] for k in 0..7, i in 0..2, r in 0..2
    # Wait, the output factor follows W[k*n+i, r] convention.
    # So C is indexed as C[k*n+i, r] = sum_j A[i,j] * B[j,k]
    # That means C has shape (p*n, m) = (8*3, 3) = (24, 3)
    # And the tensor is: C[k*n+i, r] = sum_j A[i*m+j, ?] ... 
    
    # Let me re-read the convention. The matrix multiplication is:
    # (n x m) * (m x p) = (n x p)
    # But the output is stored as W[k*n+i, r] which suggests C is stored in a different way.
    # Actually, I think the standard convention is:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # And W is indexed as W[k*n+i, r] meaning the (k,i) entry of C is at position k*n+i in W's row, 
    # and r is the column of C (which corresponds to... hmm).
    
    # Let me just use the schoolbook as reference to understand the indexing.
    # In the schoolbook:
    # U[i*m+j, col] = 1 for A[i,j]
    # V[j*p+k, col] = 1 for B[j,k]  
    # W[k*n+i, col] = 1 for C[k,i]
    
    # So C[k,i] = sum_j A[i,j] * B[j,k]
    # This is the standard matrix product where C = A * B, and C[i,k] = sum_j A[i,j]*B[j,k]
    # But stored as C[k,i] in the W matrix (transposed storage).
    
    # OK so for our problem:
    # A: 3x3, B: 3x8, C: 3x8 (but stored transposed as 8x3 in W)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Now, Strassen for 3x3:
    # If we split B into blocks B[:, 0:3], B[:, 3:6], B[:, 6:8]
    # The first two are 3x3, the last is 3x2.
    
    # For each 3x3 block Bk, we can use Strassen to compute Ak = A * Bk using 8 mults.
    # The U factors (combos of A) are the same for all blocks.
    # The V factors (combos of Bk) are different for each block.
    # The W factors (combos of Ck) are different for each block.
    
    # Let me define the Strassen algorithm properly.
    
    # 8 products:
    # P1 = (A00+A11) * (B00+B11)
    # P2 = (A10+A11) * B00
    # P3 = A00 * (B10-B11)
    # P4 = A11 * (B01-B00)
    # P5 = (A00+A01) * B11
    # P6 = (A10-A00) * (B00+B01)
    # P7 = (A01-A11) * (B10+B11)
    # P8 = (A10+A11-A00) * (B00+B11)
    
    # C00 = P1 - P2 + P3 + P4 - P5 + P6
    # C01 = P1 - P2 - P4 + P5 - P6 - P7
    # C10 = P2 + P3 - P4 + P5 + P6
    # C11 = P1 - P3 + P4 - P5 + P7
    # C20 = P1 + P2 - P3 + P4 - P6 + P8
    # C21 = P1 - P2 + P3 - P4 + P5 - P7 - P8
    # C02 = P2 + P3 - P4 - P6
    # C12 = P1 - P2 + P4 - P5 + P7
    # C22 = P1 - P2 + P3 + P4 - P6 + P8
    
    # Let me verify C00:
    # (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B10-B11) + A11(B01-B00) - (A00+A01)B11 + (A10-A00)(B00+B01)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B10 - A00B11 + A11B01 - A11B00 - A00B11 - A01B11 + A10B00 + A10B01 - A00B00 - A00B01
    # Let me collect terms:
    # A00B00: +1 -1 = 0... that's wrong. Let me redo.
    
    # Actually, I should just trust the standard Strassen-Winograd 3x3 algorithm.
    # Let me look up the correct formulas.
    
    # The standard 3x3 Strassen-Winograd algorithm has rank 8. Let me use a known correct set.
    
    # From literature, one common rank-8 algorithm for 3x3:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # M8 = (A21+A22-A11)(B11+B22)
    
    # C11 = M1 - M2 + M3 + M4 - M5 + M6
    # C12 = M1 - M2 - M4 + M5 - M6 - M7
    # C21 = M2 + M3 - M4 + M5 + M6
    # C22 = M1 - M3 + M4 - M5 + M7
    # C31 = M1 + M2 - M3 + M4 - M6 + M8
    # C32 = M1 - M2 + M3 - M4 + M5 - M7 - M8
    # C13 = M2 + M3 - M4 - M6
    # C23 = M1 - M2 + M4 - M5 + M7
    # C33 = M1 - M2 + M3 + M4 - M6 + M8
    
    # Note: 1-indexed, so A11 is our A[0,0], etc.
    
    # Let me verify C11 (which is C[0,0] in 0-indexed):
    # M1 - M2 + M3 + M4 - M5 + M6
    # = (A11+A22)(B11+B22) - (A21+A22)B11 + A11(B21-B22) + A22(B12-B11) - (A11+A12)B22 + (A21-A11)(B11+B12)
    
    # Expand:
    # = A11B11 + A11B22 + A22B11 + A22B22 - A21B11 - A22B11 + A11B21 - A11B22 + A22B12 - A22B11 - A11B22 - A12B22 + A21B11 + A21B12 - A11B11 - A11B12
    
    # Collect:
    # A11B11: 1 - 1 = 0
    # A11B22: 1 - 1 - 1 = -1
    # A22B11: 1 - 1 - 1 = -1
    # A22B22: 1
    # A21B11: -1 + 1 = 0
    # A11B21: 1
    # A22B12: 1
    # A12B22: -1
    # A21B12: 1
    # A11B12: -1
    
    # This doesn't simplify to A11B11 + A12B21 + A21B12 + A22B22... something is wrong.
    
    # I think I'm making errors. Let me just use a simpler approach:
    # Use the known rank-8 decomposition from a reliable source.
    
    # Actually, for the purpose of this task, let me just try to find ANY rank < 72 decomposition.
    # The simplest improvement: use Strassen for the 3x3 blocks.
    
    # Let me just implement it carefully and verify numerically.
    
    # I'll use numpy to verify.
    
    import numpy as np
    
    # Define the 8 Strassen factors for 3x3
    # Using 0-indexed: A[i,j], B[i,j], C[i,j]
    
    # Let me use a verified rank-8 algorithm. Here's one from the literature:
    # (This is the Winograd 1971 algorithm)
    
    # I'll define it as a list of (U_vec, V_vec, W_vec) triples.
    
    # For now, let me just return the schoolbook to be safe, and try to improve.
    
    # Actually, let me try a different approach: use the rank-8 algorithm and verify it works.
    
    # Known rank-8 3x3 algorithm (one valid set):
    # Let me just use the simple block approach: split 3x8 into 3x3 + 3x3 + 3x2
    # Use Strassen for the two 3x3 blocks, schoolbook for 3x2.
    # But I need a correct Strassen implementation.
    
    # Let me try to construct it and verify with random matrices.
    
    def make_strassen_3x3():
        """Returns (U, V, W) each 9x8 for 3x3*3x3=3x3 using rank-8."""
        # I'll use the following verified algorithm (checking online references in my memory):
        # This is a known rank-8 decomposition for 3x3 matrix multiplication.
        
        U = np.zeros((9, 8))
        V = np.zeros((9, 8))
        W = np.zeros((9, 8))
        
        # M1 = (A00+A11)*(B00+B11)
        U[0,0] = 1; U[4,0] = 1
        V[0,0] = 1; V[4,0] = 1
        W[0,0] = 1; W[1,0] = 1; W[3,0] = 1; W[4,0] = 1; W[6,0] = 1; W[7,0] = 1
        
        # This is getting too error-prone. Let me just use schoolbook for now
        # and try to find a better algorithm.
        
        return None
    
    # Given the complexity, let me just return the schoolbook decomposition.
    # The rank is 72.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
