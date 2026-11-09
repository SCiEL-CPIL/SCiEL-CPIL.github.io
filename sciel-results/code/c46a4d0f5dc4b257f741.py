"""Matrix multiplication 4x4x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """Return (U, V, W) for (2x2)*(2x2)->(2x2), rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    """
    # Standard Strassen
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1               # B11
    W[2, 1] = 1; W[3, 1] = -1 # C21, C22
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1               # A11
    V[1, 2] = 1; V[3, 2] = -1 # B12, B22
    W[1, 2] = 1; W[3, 2] = 1  # C12, C22
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1               # A22
    V[2, 3] = 1; V[0, 3] = -1 # B21, B11
    W[0, 3] = 1; W[2, 3] = 1  # C11, C21
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1               # B22
    W[0, 4] = -1; W[1, 4] = 1 # C11, C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1 # A21, A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1               # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1 # A12, A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1               # C11
    
    return U, V, W

def _strassen_2x2x3():
    """Return (U, V, W) for (2x2)*(2x3)->(2x3), rank 14.
    U shape (4,14), V shape (6,14), W shape (6,14).
    n=2, m=2, p=3.
    """
    # Split B into two 2x2 and 2x1 blocks? Or use Strassen structure.
    # For (2x2)*(2x3), we can view as two (2x2)*(2x2) multiplications plus (2x2)*(2x1).
    # Actually, let's use the standard approach: split the 3 columns into 2+1.
    # M = A * [B1 | B2] where B1 is 2x2, B2 is 2x1.
    # C1 = A*B1 (rank 7 via Strassen), C2 = A*B2 (rank 4 via schoolbook for 2x2*2x1).
    # Total rank = 7 + 4 = 11.
    # Wait, 2x2 * 2x1: A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # This needs 4 multiplications (schoolbook).
    # So total rank = 7 + 4 = 11.
    
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Part 1: Strassen for A * B1 (2x2 * 2x2 -> 2x2)
    U1, V1, W1 = _strassen_2x2x2()
    U[:, :7] = U1
    V[:, :7] = V1
    # W1 is (4,7) for C1 which is 2x2. C1 indices: C[0,0], C[0,1], C[1,0], C[1,1]
    # In full C (2x3), C1 occupies columns 0,1.
    # W[k*n+i, c] with n=2. C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3].
    # But wait, for the full problem, W is (p*n, rank) = (6, rank).
    # C is 2x3. C[i,k] is at index k*n+i in W.
    # C1 is the first 2 columns of C.
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]  -- wait, no.
    # Let me re-read the convention: W[k*n+i, r].
    # So C[i,k] is at row k*n+i.
    # C[0,0] -> row 0*2+0 = 0
    # C[0,1] -> row 1*2+0 = 2
    # C[0,2] -> row 2*2+0 = 4
    # C[1,0] -> row 0*2+1 = 1
    # C[1,1] -> row 1*2+1 = 3
    # C[1,2] -> row 2*2+1 = 5
    
    # In _strassen_2x2x2, W1 is (4,7) with n=2, p=2.
    # W1[k*2+i, c] corresponds to C1[i,k] for k=0,1.
    # W1[0, c] = C1[0,0]
    # W1[1, c] = C1[1,0]
    # W1[2, c] = C1[0,1]
    # W1[3, c] = C1[1,1]
    
    # So for the full W (6, 11):
    # C[0,0] is W_full[0]
    # C[1,0] is W_full[1]
    # C[0,1] is W_full[2]
    # C[1,1] is W_full[3]
    
    W[0, :7] = W1[0, :]  # C[0,0]
    W[1, :7] = W1[1, :]  # C[1,0]
    W[2, :7] = W1[2, :]  # C[0,1]
    W[3, :7] = W1[3, :]  # C[1,1]
    
    # Part 2: Schoolbook for A * B2 (2x2 * 2x1 -> 2x1)
    # B2 is columns 2 of the original B (2x3).
    # B indices: B[0,2] = V[2*2+0] = V[4], B[1,2] = V[5].
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Mult 7: A[0,0] * B[0,2]
    U[0, 7] = 1
    V[4, 7] = 1
    W[4, 7] = 1  # C[0,2] -> W[2*2+0] = W[4]
    
    # Mult 8: A[0,1] * B[1,2]
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1  # C[0,2]
    
    # Mult 9: A[1,0] * B[0,2]
    U[2, 9] = 1
    V[4, 9] = 1
    W[5, 9] = 1  # C[1,2] -> W[2*2+1] = W[5]
    
    # Mult 10: A[1,1] * B[1,2]
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1  # C[1,2]
    
    return U, V, W

def _strassen_2x3x2():
    """Return (U, V, W) for (2x3)*(3x2)->(2x2), rank 14.
    U shape (6,14), V shape (6,14), W shape (4,14).
    n=2, m=3, p=2.
    """
    # Split A into (2x2 + 2x1) and B into (2x2 + 1x2).
    # Actually, let's use a different decomposition.
    # (2x3)*(3x2): 
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x2, B2 is 1x2.
    # C = A1*B1 + A2*B2.
    # A1*B1 is 2x2*2x2 -> rank 7 (Strassen).
    # A2*B2 is 2x1*1x2 -> rank 1 (outer product).
    # Total rank = 7 + 1 = 8.
    
    U = np.zeros((6, 8))
    V = np.zeros((6, 8))
    W = np.zeros((4, 8))
    
    # Part 1: Strassen for A1 * B1 (2x2 * 2x2 -> 2x2)
    U1, V1, W1 = _strassen_2x2x2()
    # A1 is A[:, 0:2] -> U indices 0,1,2,3 (i*m+j, m=3, so i*3+j)
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # B1 is B[0:2, :] -> V indices 0,1,2,3 (j*p+k, p=2, so j*2+k)
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # C is 2x2 -> W indices 0,1,2,3 (k*n+i, n=2, so k*2+i)
    # C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3]
    
    # Map Strassen U (4,7) to U (6,8)
    # Strassen U row 0 -> A11 -> U[0]
    # Strassen U row 1 -> A12 -> U[1]
    # Strassen U row 2 -> A21 -> U[3]
    # Strassen U row 3 -> A22 -> U[4]
    U[0, :7] = U1[0, :]
    U[1, :7] = U1[1, :]
    U[3, :7] = U1[2, :]
    U[4, :7] = U1[3, :]
    
    # Map Strassen V (4,7) to V (6,8)
    # Strassen V row 0 -> B11 -> V[0]
    # Strassen V row 1 -> B12 -> V[1]
    # Strassen V row 2 -> B21 -> V[2]
    # Strassen V row 3 -> B22 -> V[3]
    V[0, :7] = V1[0, :]
    V[1, :7] = V1[1, :]
    V[2, :7] = V1[2, :]
    V[3, :7] = V1[3, :]
    
    # W1 is (4,7) for C1 (2x2).
    W[:, :7] = W1
    
    # Part 2: A2 * B2 (2x1 * 1x2 -> 2x2)
    # A2 is A[:, 2] -> U[2] (A[0,2]), U[5] (A[1,2])
    # B2 is B[2, :] -> V[4] (B[2,0]), V[5] (B[2,1])
    # C = A2 * B2
    # C[0,0] = A[0,2]*B[2,0]
    # C[0,1] = A[0,2]*B[2,1]
    # C[1,0] = A[1,2]*B[2,0]
    # C[1,1] = A[1,2]*B[2,1]
    
    # Mult 7: A[0,2] * B[2,0]
    U[2, 7] = 1
    V[4, 7] = 1
    W[0, 7] = 1  # C[0,0]
    
    # But wait, we need 4 multiplications for the outer product.
    # Actually, A2 is 2x1, B2 is 1x2. The product is 2x2.
    # C[i,k] = A[i,2] * B[2,k].
    # This is 4 terms, but they share factors.
    # Can we do it in fewer? No, it's a rank-4 tensor (outer product of 2x1 and 1x2).
    # Wait, the tensor for (2x1)*(1x2)->(2x2) has rank 4?
    # C[0,0] = A0*B0
    # C[0,1] = A0*B1
    # C[1,0] = A1*B0
    # C[1,1] = A1*B1
    # This is the tensor of rank 4 (it's the product of two 2-dimensional spaces).
    # So we need 4 more multiplications. Total rank = 7 + 4 = 11.
    
    # Let me redo with rank 11.
    U = np.zeros((6, 11))
    V = np.zeros((6, 11))
    W = np.zeros((4, 11))
    
    U[0, :7] = U1[0, :]
    U[1, :7] = U1[1, :]
    U[3, :7] = U1[2, :]
    U[4, :7] = U1[3, :]
    V[0, :7] = V1[0, :]
    V[1, :7] = V1[1, :]
    V[2, :7] = V1[2, :]
    V[3, :7] = V1[3, :]
    W[:, :7] = W1
    
    # Mult 7: A[0,2] * B[2,0]
    U[2, 7] = 1
    V[4, 7] = 1
    W[0, 7] = 1  # C[0,0]
    
    # Mult 8: A[0,2] * B[2,1]
    U[2, 8] = 1
    V[5, 8] = 1
    W[2, 8] = 1  # C[0,1]
    
    # Mult 9: A[1,2] * B[2,0]
    U[5, 9] = 1
    V[4, 9] = 1
    W[1, 9] = 1  # C[1,0]
    
    # Mult 10: A[1,2] * B[2,1]
    U[5, 10] = 1
    V[5, 10] = 1
    W[3, 10] = 1  # C[1,1]
    
    return U, V, W

def _strassen_2x3x3():
    """Return (U, V, W) for (2x3)*(3x3)->(2x3), rank 22.
    U shape (6,22), V shape (9,22), W shape (6,22).
    n=2, m=3, p=3.
    """
    # Split into (2x2)*(2x2) + (2x1)*(2x2) + (2x2)*(2x1) + (2x1)*(2x1)?
    # Or use Strassen on the 2x2 blocks and schoolbook on the rest.
    # Let's use the approach from the knowledge: Strassen(7)+schoolbook(4) per (2x2)*(2x3) block.
    # Wait, the knowledge says: "Replaces schoolbook rank-48 for the 4x4x3 subproblem with a rank-44 decomposition using Strassen(7)+schoolbook(4) per (2x2)*(2x3) block"
    # So for (2x2)*(2x3), rank is 7+4=11.
    # For (2x3)*(3x3), we can split B into (2x2 + 2x1) columns? No, B is 3x3.
    # Let's split A into (2x2 + 2x1) and B into (2x2 + 1x2) rows?
    # Actually, let's just use a direct construction.
    # (2x3)*(3x3): 
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x3, B2 is 1x3.
    # C = A1*B1 + A2*B2.
    # A1*B1 is (2x2)*(2x3) -> rank 11 (from _strassen_2x2x3).
    # A2*B2 is (2x1)*(1x3) -> rank 6 (outer product).
    # Total rank = 11 + 6 = 17.
    
    U = np.zeros((6, 17))
    V = np.zeros((9, 17))
    W = np.zeros((6, 17))
    
    # Part 1: A1 * B1 (2x2 * 2x3 -> 2x3)
    U1, V1, W1 = _strassen_2x2x3()
    # A1 is A[:, 0:2] -> U[0], U[1], U[3], U[4]
    # B1 is B[0:2, :] -> V[0], V[1], V[2], V[3], V[4], V[5]
    # C is 2x3 -> W[0], W[1], W[2], W[3], W[4], W[5]
    
    U[0, :11] = U1[0, :]
    U[1, :11] = U1[1, :]
    U[3, :11] = U1[2, :]
    U[4, :11] = U1[3, :]
    V[0, :11] = V1[0, :]
    V[1, :11] = V1[1, :]
    V[2, :11] = V1[2, :]
    V[3, :11] = V1[3, :]
    V[4, :11] = V1[4, :]
    V[5, :11] = V1[5, :]
    W[:, :11] = W1
    
    # Part 2: A2 * B2 (2x1 * 1x3 -> 2x3)
    # A2 is A[:, 2] -> U[2], U[5]
    # B2 is B[2, :] -> V[6], V[7], V[8]
    # C[i,k] = A[i,2] * B[2,k]
    
    for i in range(2):
        for k in range(3):
            col = 11 + i * 3 + k
            U[i * 3 + 2, col] = 1  # A[i,2]
            V[6 + k, col] = 1      # B[2,k]
            W[k * 2 + i, col] = 1  # C[i,k]
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 7
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block structure:
    # A is 4x4, B is 4x7.
    # Split A into 2x2 blocks: A11, A12, A21, A22.
    # Split B into 2x7 blocks: B1, B2 (each 2x7).
    # C = A*B is 4x7.
    # C1 = A1*B1 (2x7), C2 = A2*B2 (2x7).
    # C1 = A11*B11 + A12*B21
    # C2 = A21*B12 + A22*B22
    
    # B1 is 2x7, B2 is 2x7.
    # B11 is 2x4 (first 4 cols of B1), B21 is 2x3 (last 3 cols of B1)?
    # Wait, B is 4x7. Split rows into 2+2.
    # B1 = B[0:2, :] is 2x7.
    # B2 = B[2:4, :] is 2x7.
    
    # C1 = A1 * B1 where A1 is 2x4, B1 is 2x7? No, A1 is 2x4? No, A is 4x4.
    # A1 = A[0:2, :] is 2x4.
    # B1 = B[0:2, :] is 2x7.
    # C1 = A1 * B1 is 2x7.
    # But A1 is 2x4 and B1 is 2x7. The inner dimension is 2? No, A is 4x4, B is 4x7.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # C1[i,k] = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k].
    # Let A1 = A[0:2, 0:2] (2x2), A2 = A[0:2, 2:4] (2x2).
    # Let B1 = B[0:2, :] (2x7), B2 = B[2:4, :] (2x7).
    # C1 = A1*B1 + A2*B2.
    
    # So for each 2x2 block of A, we multiply by the corresponding 2x7 block of B.
    # A11 (2x2) * B1 (2x7) -> (2x7)
    # A12 (2x2) * B2 (2x7) -> (2x7)
    # A21 (2x2) * B1 (2x7) -> (2x7)
    # A22 (2x2) * B2 (2x7) -> (2x7)
    
    # We need a decomposition for (2x2)*(2x7).
    # Split B (2x7) into B1 (2x4) + B2 (2x3).
    # (2x2)*(2x4) -> rank 11 (from _strassen_2x2x3? No, that was 2x2*2x3).
    # Let's make _strassen_2x2x4 and _strassen_2x2x3.
    
    def _strassen_2x2x4():
        # (2x2)*(2x4) -> (2x4), rank 11 + 11 = 22? No.
        # Split B into two 2x2 blocks.
        # (2x2)*(2x2) -> rank 7.
        # Two such blocks -> rank 14.
        U = np.zeros((4, 14))
        V = np.zeros((8, 14))
        W = np.zeros((8, 14))
        U1, V1, W1 = _strassen_2x2x2()
        U[:, :7] = U1
        V[0:4, :7] = V1
        # W1 is (4,7) for C1 (2x2). C1 is columns 0,1 of C (2x4).
        # C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3].
        W[0, :7] = W1[0, :]
        W[1, :7] = W1[1, :]
        W[2, :7] = W1[2, :]
        W[3, :7] = W1[3, :]
        
        U[:, 7:] = U1
        V[4:8, 7:] = V1
        W[4, 7:] = W1[0, :]
        W[5, 7:] = W1[1, :]
        W[6, 7:] = W1[2, :]
        W[7, 7:] = W1[3, :]
        return U, V, W
    
    def _strassen_2x2x3():
        # (2x2)*(2x3) -> (2x3), rank 11.
        return _strassen_2x2x3()
    
    # Total rank = 4 * (rank of (2x2)*(2x7))
    # (2x2)*(2x7) = (2x2)*(2x4) + (2x2)*(2x3) -> rank 14 + 11 = 25.
    # Total rank = 4 * 25 = 100.
    # Schoolbook is 4*4*7 = 112.
    # This is an improvement.
    
    rank_total = 4 * 25
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    blocks = [
        (0, 0, 0, 0),  # A11, B1
        (0, 2, 2, 0),  # A12, B2
        (2, 0, 0, 2),  # A21, B1
        (2, 2, 2, 2),  # A22, B2
    ]
    
    for bi, (ai, aj, bi_row, bi_col) in enumerate(blocks):
        offset = bi * 25
        # A block is at rows ai:ai+2, cols aj:aj+2.
        # B block is at rows bi_row:bi_row+2, cols 0:7.
        # C block is at rows ai:ai+2, cols 0:7.
        
        # (2x2)*(2x7)
        U_block, V_block, W_block = _strassen_2x2x7()
        
        # Map U_block (4,25) to U (16,100)
        for i in range(2):
            for j in range(2):
                U[(ai + i) * 4 + (aj + j), offset:offset+25] = U_block[i * 2 + j, :]
        
        # Map V_block (14,25) to V (28,100)
        for j in range(2):
            for k in range(7):
                V[(bi_row + j) * 7 + k, offset:offset+25] = V_block[j * 7 + k, :]
        
        # Map W_block (28,25) to W (28,100)
        for k in range(7):
            for i in range(2):
                W[k * 4 + (ai + i), offset:offset+25] = W_block[k * 2 + i, :]
    
    # Convert to lists of ints/floats
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _strassen_2x2x7():
    """Return (U, V, W) for (2x2)*(2x7)->(2x7), rank 25.
    U shape (4,25), V shape (14,25), W shape (14,25).
    n=2, m=2, p=7.
    """
    # Split B (2x7) into B1 (2x4) + B2 (2x3).
    # (2x2)*(2x4) -> rank 14.
    # (2x2)*(2x3) -> rank 11.
    # Total rank = 25.
    
    U = np.zeros((4, 25))
    V = np.zeros((14, 25))
    W = np.zeros((14, 25))
    
    # Part 1: (2x2)*(2x4)
    U1, V1, W1 = _strassen_2x2x4()
    U[:, :14] = U1
    V[0:8, :14] = V1
    W[0:8, :14] = W1
    
    # Part 2: (2x2)*(2x3)
    U2, V2, W2 = _strassen_2x2x3()
    U[:, 14:] = U2
    V[8:14, 14:] = V2
    W[8:14, 14:] = W2
    
    return U, V, W

def _strassen_2x2x4():
    """Return (U, V, W) for (2x2)*(2x4)->(2x4), rank 14.
    U shape (4,14), V shape (8,14), W shape (8,14).
    n=2, m=2, p=4.
    """
    # Split B (2x4) into two (2x2) blocks.
    U = np.zeros((4, 14))
    V = np.zeros((8, 14))
    W = np.zeros((8, 14))
    
    U1, V1, W1 = _strassen_2x2x2()
    U[:, :7] = U1
    V[0:4, :7] = V1
    W[0:4, :7] = W1
    
    U[:, 7:] = U1
    V[4:8, 7:] = V1
    W[4:8, 7:] = W1
    
    return U, V, W

def _strassen_2x2x2():
    """Return (U, V, W) for (2x2)*(2x2)->(2x2), rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    n=2, m=2, p=2.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_2x2x3():
    """Return (U, V, W) for (2x2)*(2x3)->(2x3), rank 11.
    U shape (4,11), V shape (6,11), W shape (6,11).
    n=2, m=2, p=3.
    """
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    U1, V1, W1 = _strassen_2x2x2()
    U[:, :7] = U1
    V[:, :7] = V1
    W[0, :7] = W1[0, :]
    W[1, :7] = W1[1, :]
    W[2, :7] = W1[2, :]
    W[3, :7] = W1[3, :]
    
    U[0, 7] = 1; V[4, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[2, 9] = 1; V[4, 9] = 1; W[5, 9] = 1
    U[3, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    
    return U, V, W
