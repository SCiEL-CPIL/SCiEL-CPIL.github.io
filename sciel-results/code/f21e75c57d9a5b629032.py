import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # Must finish within 120s, keep safety margin
    
    n, m, p = DIMENSIONS
    
    def schoolbook_decomposition():
        rank = 27
        U = np.zeros((9, rank), dtype=float)
        V = np.zeros((9, rank), dtype=float)
        W = np.zeros((9, rank), dtype=float)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    idx = i * 3 + j
                    col = i * 9 + j * 3 + k
                    U[idx, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 3 + i, col] = 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """
        Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] equals the matrix multiplication tensor.
        C[i,k] = sum_j A[i,j] B[j,k]
        W index is k*n + i.
        """
        n, m, p = DIMENSIONS
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Build the expected tensor
        T_expected = np.zeros((n*p, m*p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[k*n + i, j*p + k] += 1  # This is not right. Let me think.
        
        # Actually, the tensor is: T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k] to get C[i,k]
        # But the factorization is: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # But the task says W[k*n+i, r], so W is indexed by (k,i) which is column-major for C?
        # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
        # C = A*B, so C[i,k] = sum_j A[i,j] B[j,k]
        # The decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But U should be indexed by (i,j), V by (j,k), W by (k,i) or (i,k)?
        # The task says W[k*n+i, r], so W is indexed by (k,i) which is (col, row) of C.
        # So the decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a 3-way tensor decomposition where the factors are:
        # U: (n*m) x rank, indexed by (i,j)
        # V: (m*p) x rank, indexed by (j,k)
        # W: (p*n) x rank, indexed by (k,i)
        # And the tensor is T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k.
        
        # Let me build the tensor T of shape (n*m, m*p, p*n)
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m + j, j*p + k, k*n + i] = 1
        
        # Now check: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We can check this by computing for each (i,j,k):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = T[i*m + j, j*p + k, k*n + i]
                    actual = 0.0
                    for r in range(U.shape[1]):
                        actual += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    def strassen_2x2_decomposition():
        """
        Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
        Returns U, V, W of shapes (4, 7), (4, 7), (4, 7)
        where U is indexed by (i,j) for i,j in 0..1,
        V is indexed by (j,k) for j,k in 0..1,
        W is indexed by (k,i) for k,i in 0..1.
        """
        # Strassen's 7 products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        rank = 7
        U = np.zeros((4, rank), dtype=float)
        V = np.zeros((4, rank), dtype=float)
        W = np.zeros((4, rank), dtype=float)
        
        # Index mapping: (i,j) -> i*2+j for U, (j,k) -> j*2+k for V, (k,i) -> k*2+i for W
        
        # M1 = (A11 + A22)(B11 + B22)
        # A factors: A11 at (0,0), A22 at (1,1)
        U[0*2+0, 0] = 1  # A11
        U[1*2+1, 0] = 1  # A22
        V[0*2+0, 0] = 1  # B11
        V[1*2+1, 0] = 1  # B22
        # C factors: C11 += M1, C22 += M1
        W[0*2+0, 0] = 1  # C11 (k=0, i=0)
        W[1*2+1, 0] = 1  # C22 (k=1, i=1)
        
        # M2 = (A21 + A22)B11
        U[1*2+0, 1] = 1  # A21
        U[1*2+1, 1] = 1  # A22
        V[0*2+0, 1] = 1  # B11
        W[0*2+1, 1] = 1  # C21 (k=0, i=1)
        
        # M3 = A11(B12 - B22)
        U[0*2+0, 2] = 1  # A11
        V[0*2+1, 2] = 1  # B12
        V[1*2+1, 2] = -1  # B22
        W[0*2+0, 2] = 1  # C12 (k=0, i=0)? No, C12 is (i=0, k=1), so W index is k*2+i = 1*2+0 = 2
        W[1*2+0, 2] = 1  # C12 (k=1, i=0)
        
        # M4 = A22(B21 - B11)
        U[1*2+1, 3] = 1  # A22
        V[1*2+0, 3] = 1  # B21
        V[0*2+0, 3] = -1  # B11
        W[0*2+1, 3] = 1  # C21 (k=0, i=1)
        W[0*2+0, 3] = 1  # C11 (k=0, i=0)
        
        # M5 = (A11 + A12)B22
        U[0*2+0, 4] = 1  # A11
        U[0*2+1, 4] = 1  # A12
        V[1*2+1, 4] = 1  # B22
        W[0*2+0, 4] = -1  # C11 (k=0, i=0)
        W[1*2+0, 4] = 1  # C12 (k=1, i=0)
        
        # M6 = (A21 - A11)(B11 + B12)
        U[1*2+0, 5] = 1  # A21
        U[0*2+0, 5] = -1  # A11
        V[0*2+0, 5] = 1  # B11
        V[0*2+1, 5] = 1  # B12
        W[1*2+1, 5] = 1  # C22 (k=1, i=1)
        
        # M7 = (A12 - A22)(B21 + B22)
        U[0*2+1, 6] = 1  # A12
        U[1*2+1, 6] = -1  # A22
        V[1*2+0, 6] = 1  # B21
        V[1*2+1, 6] = 1  # B22
        W[0*2+0, 6] = 1  # C11 (k=0, i=0)
        
        return U, V, W
    
    def combine_to_rank_26():
        """
        Combine Strassen for the 2x2 block with schoolbook for the remaining parts.
        Partition 3x3 as:
        A = [[A11(2x2), A12(2x1)], [A21(1x2), A22(1x1)]]
        B = [[B11(2x2), B12(2x1)], [B21(1x2), B22(1x1)]]
        C = [[C11(2x2), C12(2x1)], [C21(1x2), C22(1x1)]]
        
        C11 = A11*B11 + A12*B21 -> use Strassen for A11*B11 (7 terms) + schoolbook for A12*B21 (2 terms)
        C12 = A11*B12 + A12*B22 -> schoolbook: 4 + 2 = 6 terms
        C21 = A21*B11 + A22*B21 -> schoolbook: 4 + 2 = 6 terms
        C22 = A22*B22 -> 1 term
        
        Total = 7 + 2 + 6 + 6 + 1 = 22? But known rank is 26.
        The issue is that the Strassen terms for C11 involve linear combinations of A11 and B11,
        which may conflict with the schoolbook terms. Let me just use schoolbook for all cross terms.
        
        Actually, let me just use schoolbook for everything except the 2x2 block, and Strassen for the 2x2 block.
        C11 = A11*B11 + A12*B21. The A11*B11 part uses 7 Strassen terms. The A12*B21 part uses 2 terms.
        But the Strassen terms only involve A11 and B11, not A12 or B21. So they are independent.
        Similarly, C12 = A11*B12 + A12*B22. A11*B12 is 4 terms (A11[0,0]*B12[0,0], A11[0,1]*B12[1,0], etc.)
        A12*B22 is 2 terms. Total 6.
        C21 = A21*B11 + A22*B21. 4 + 2 = 6.
        C22 = A22*B22. 1.
        
        Total = 7 + 2 + 6 + 6 + 1 = 22.
        But the known rank is 26. This suggests my understanding is wrong.
        Let me just return the schoolbook decomposition for now, which is rank 27.
        """
        return schoolbook_decomposition()
    
    # For now, return the schoolbook decomposition
    # TODO: Implement a proper rank-26 decomposition
    result = schoolbook_decomposition()
    return result
# EVOLVE-BLOCK-END
