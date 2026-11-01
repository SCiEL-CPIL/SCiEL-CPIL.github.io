import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 10, 25, 10
    
    start_time = time.time()
    deadline = 140.0
    
    # Build rank-40 decomposition using Strassen for 4x4 core + 5 cross terms
    # Total rank: 2 * (15 + 5) = 40
    
    rank = 40
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # For each output row i (i=0,1), we have 20 terms
    # Terms 0-14: Strassen 4x4 for columns k=0..3
    # Terms 15-19: Cross terms involving index 4
    
    # Strassen 4x4 decomposition (for i in {0,1}, k in {0,1,2,3})
    # M1 = (A_{i,0}+A_{i,2})(B_{0,k}+B_{2,k})
    # M2 = (A_{i,2}+A_{i,3})B_{0,k}
    # M3 = A_{i,0}(B_{2,k}-B_{3,k})
    # M4 = A_{i,3}(B_{1,k}-B_{0,k})
    # M5 = (A_{i,0}+A_{i,2})B_{3,k}
    # M6 = (B_{0,k}+B_{3,k})(A_{i,2}-A_{i,0})
    # M7 = (B_{1,k}-B_{3,k})(A_{i,1}+A_{i,2})
    
    # Wait, let me use the standard 4x4 Strassen with blocks of size 2x2
    # Actually, for 4x4, we can use Strassen recursively or directly.
    # Let me use a direct 4x4 rank-15 decomposition.
    
    # Standard 4x4 Strassen (treating 4x4 as 2x2 blocks of 2x2):
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A10)B11
    # P6 = (B00+B11)(A10-A00)
    # P7 = (B01-B11)(A01+A10)
    # C00 = P1+P4-P5-P6
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P7
    
    # For 4x4 with indices 0,1,2,3:
    # A block: rows 0,1 vs rows 2,3; B block: cols 0,1 vs cols 2,3
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4], C10 = C[2:4, 0:2], C11 = C[2:4, 2:4]
    
    # For our 5x5, we split into 4x4 core (indices 0-3) and index 4
    # The 4x4 core uses Strassen (rank 15)
    # The cross terms (involving index 4) need 5 additional rank-1 terms per row
    
    # Let me define the 7 Strassen products for 4x4:
    # For output row i (i=0 or 1), the 4x4 part involves A[i, 0:4] and B[0:4, :]
    
    # Actually, let me think differently. The tensor is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For i=0: C[0,k] = sum_j A[0,j] B[j,k]
    # For i=1: C[1,k] = sum_j A[1,j] B[j,k]
    
    # Split j into {0,1,2,3} and {4}:
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k] + A[i,4] B[4,k]
    
    # For the 4x4 part (j in {0,1,2,3}, k in {0,1,2,3}): use Strassen rank 15
    # For k=4: C[i,4] = sum_{j=0}^{3} A[i,j] B[j,4] + A[i,4] B[4,4]
    #   This needs 4 rank-1 terms for j=0..3 plus 1 for j=4: total 5
    # For i=4... wait, n=2, so i is only 0,1. p=5, so k is 0..4.
    # But the output is C[i,k] with i in {0,1}, k in {0,1,2,3,4}
    # Wait, no: A is 2x5 (n=2 rows, m=5 cols), B is 5x5 (m=5 rows, p=5 cols)
    # C is 2x5 (n=2 rows, p=5 cols)
    # C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    
    # So for each i in {0,1}:
    #   C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k] + A[i,4] B[4,k]
    
    # For k in {0,1,2,3}: the 4x4 submatrix C[i, 0:4] = A[i, 0:4] @ B[0:4, 0:4]
    #   This is a 1x4 times 4x4 = 1x4, not a 4x4 multiplication.
    #   
    # Hmm, I need to reconsider. The "4x4 core" idea doesn't directly apply here
    # because we're multiplying a 2x5 by 5x5, not 5x5 by 5x5.
    
    # Let me reconsider the approach. The tensor has structure:
    # T[i,j,k] = 1 if the product contributes to C[i,k]
    # For fixed i, C[i,k] = sum_j A[i,j] B[j,k]
    
    # A rank-1 term is: u_i * v_j * w_k where the term contributes to C[i,k]
    # via A[i,j]*B[j,k]
    
    # So a rank-1 term with U[i,j]=a, V[j,k]=b, W[i,k]=c contributes a*b*c to C[i,k]
    # We need sum of such products = 1 for each (i,k) pair.
    
    # For 2x5x5, there are 2*5 = 10 output entries.
    # Schoolbook: 2*5*5 = 50 terms.
    
    # Strassen-like: For the 5x5 B matrix, we can group columns.
    # Let's try: split B's columns into groups and use Strassen-style additions.
    
    # Alternative: Use the fact that for each i, we're computing a row of A times B.
    # C[i,:] = A[i,:] @ B
    # This is a 1x5 times 5x5 = 1x5 matrix product.
    # The rank of the 1x5x5 tensor is 5 (schoolbook).
    # But we can do better with Strassen-like methods?
    
    # Actually, for 1xn x nxn, the rank is n (you need at least n terms).
    # So for each i, we need at least 5 terms, total at least 10.
    # But the Strassen approach might help by sharing structure between i=0 and i=1.
    
    # Let me try a different approach: use the 5x5 matrix multiplication rank.
    # The 5x5 matrix multiplication tensor has rank at most 25 (schoolbook).
    # Strassen for 4x4 gives rank 15 for 4x4.
    # For 5x5, we can use 4x4 Strassen + 5 cross terms = 20.
    
    # But our problem is 2x5x5, not 5x5x5.
    
    # Let me just try to find a good decomposition computationally.
    
    # Approach: Start with schoolbook and try to merge terms.
    # Or use a greedy algorithm.
    
    def verify(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    # C[i,k] should be sum_j A[i,j]*B[j,k]
                    # In tensor form: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    pass
                # Actually, the tensor T[i,j,k] = 1 for all i,j,k
                # And the decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1
                s = 0.0
                for c in range(r):
                    s += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
        return True
    
    # Let me try a smarter construction.
    # For 2x5x5, let's use the following structure:
    # Group the 5 columns of B into {0,1,2,3} and {4}
    # For k in {0,1,2,3}: use a 4-column Strassen-like approach
    # For k=4: use 5 terms
    
    # Actually, let me try to use known results.
    # The rank of 2x5x5 matrix multiplication is at most 40 (using 5x5 rank 20 per row).
    
    # Let me construct a rank-40 decomposition:
    # For each i in {0,1}, use a 5x5 matrix multiplication decomposition of rank 20
    # for computing C[i,:] = A[i,:] @ B
    
    # 5x5 matrix multiplication (1x5 times 5x5) rank 20:
    # Split the 5x5 into 4x4 core + cross terms
    # 4x4 part: rank 15 (Strassen)
    # Cross terms: 5 (for the 5th row/col)
    # Total: 20
    
    # For 1x5 times 5x5:
    # C[k] = sum_j A[j] B[j,k]
    # Split j into {0,1,2,3} and {4}
    # C[k] = sum_{j=0}^{3} A[j] B[j,k] + A[4] B[4,k]
    
    # For k in {0,1,2,3}: C[k] involves B[0:4, k]
    # For k=4: C[4] = sum_{j=0}^{3} A[j] B[j,4] + A[4] B[4,4]
    
    # The 4x4 part (j in {0,1,2,3}, k in {0,1,2,3}):
    # This is a 1x4 times 4x4 = 1x4 product.
    # Rank of 1x4x4 tensor is 4 (schoolbook).
    # Can we do better? No, the rank is exactly 4 for 1xn x nxn.
    
    # So for the 4x4 part, we need 4 terms per i.
    # For k=4, we need 5 terms per i.
    # Total per i: 4 + 5 = 9 terms.
    # Total for 2 rows: 18 terms.
    
    # But this doesn't share structure between i=0 and i=1.
    
    # To share, we need to use the fact that both rows use the same B.
    # The 2x5x5 tensor can be seen as two 1x5x5 tensors sharing the B factor.
    
    # Let me try a different decomposition:
    # Use Strassen on the 2x2 block structure of A.
    # A is 2x5, so we can't really do 2x2 blocks.
    
    # Let me just try to find a rank-40 or lower decomposition computationally.
    
    # Greedy approach: start with schoolbook and try to eliminate terms.
    
    # Actually, let me try a specific construction:
    # For the 5x5 B matrix, use a rank-20 decomposition.
    # The 5x5 matrix multiplication tensor (5x5 times 5x5) has rank at most 20.
    # But we only need 1x5 times 5x5.
    
    # Let me try: rank = 40, and see if we can verify it.
    
    # Construction:
    # For i=0:
    #   Terms 0-14: Strassen 4x4 for C[0, 0:4]
    #   Terms 15-19: Cross terms for C[0, 4]
    # For i=1:
    #   Terms 20-34: Strassen 4x4 for C[1, 0:4]
    #   Terms 35-39: Cross terms for C[1, 4]
    
    # But the Strassen 4x4 is for 4x4 times 4x4, not 1x4 times 4x4.
    
    # I think the key insight is that we should use the 5x5 matrix multiplication
    # decomposition and then restrict to our case.
    
    # Let me try a completely different approach: use the known rank bound.
    # The rank of 2x5x5 matrix multiplication is at most 40.
    # Let me just return a schoolbook decomposition of rank 50 for now,
    # and try to improve it.
    
    # Actually, let me try to use a rank-40 decomposition by exploiting
    # the structure more carefully.
    
    # New idea: Use the fact that C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as:
    # C[i,k] = sum_j A[i,j] (B[j,k])
    # 
    # If we precompute some combinations of A's rows, we can reduce the rank.
    # For example, let S0 = A[0,0] + A[0,1] + A[0,2] + A[0,3]
    # and T0 = B[0,0] + B[1,0] + B[2,0] + B[3,0]
    # Then S0 * T0 = sum_{j=0}^{3} sum_{k=0}^{3} A[0,j] B[k,0]
    # This isn't quite what we want.
    
    # Let me try yet another approach: use the tensor's symmetry.
    # The tensor T[i,j,k] = 1 for all i,j,k is symmetric in a certain sense.
    
    # For now, let me return a rank-40 decomposition that I construct explicitly.
    
    # Rank-40 construction:
    # For each i in {0,1}, use 20 terms:
    #   15 terms for the 4x4 core (using Strassen)
    #   5 terms for the cross couplings with index 4
    
    # Strassen 4x4 (for computing a 4x4 matrix product):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (B00+B11)(A10-A00)
    # M7 = (B01-B11)(A01+A10)
    
    # For 1x4 times 4x4, we only need the output row, not the full 4x4.
    # So we can simplify.
    
    # Let me just try to find a valid decomposition computationally.
    
    # Fallback: schoolbook rank 50
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
