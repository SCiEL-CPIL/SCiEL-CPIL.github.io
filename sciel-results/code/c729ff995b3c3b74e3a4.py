"""Program to find a rank-minimal exact tensor decomposition for 2x4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    # Tensor T of shape (n, m, p) where T[i, j, k] = 1 if this corresponds to element (i,k) of result from (i,j) of A and (j,k) of B
    # Actually, the matrix multiplication tensor T[i, j, k] represents the contribution of A[i,j] * B[j,k] to C[i,k]
    # So T[i, j, k] = 1 for all valid i,j,k
    
    # The output convention is W[k*n+i, r]
    # Input U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The product sum_r U[i*m+j, r] * V[j*p+k, r] should equal C[i,k]
    # And C[i,k] = sum_j A[i,j] * B[j,k]
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] = sum_j delta(j,j) = 1 if this is the (i,k) term
    # Actually, the tensor decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But wait, the task says "exactly decompose the matrix multiplication tensor"
    # The standard tensor decomposition for matrix multiplication is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 (since each (i,j,k) triple contributes exactly once)
    
    # Let's use Strassen's algorithm for 2x2x2 and extend it.
    # For 2x4x4, we can view it as:
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into [A1, A2] where A1, A2 are 2x2
    # Split B into [[B1, B2], [B3, B4]] where each is 2x2
    # Then C = [A1, A2] * [[B1, B2], [B3, B4]] = [A1*B1 + A2*B3, A1*B2 + A2*B4]
    
    # This is not directly Strassen. Let me think differently.
    # 2x4x4 can be seen as two 2x2x4 products or use a different approach.
    
    # Actually, let's use the known rank 28 decomposition.
    # The rank of M_{2,4,4} is known to be 28.
    # Let me construct it using Strassen on a 4x4x4 by embedding.
    
    # Alternative: use the fact that M_{2,4,4} can be decomposed by splitting the 4 dimension.
    # A (2x4) * B (4x4) = C (2x4)
    # Split B into B_left (4x2) and B_right (4x2)
    # C_left (2x2) = A * B_left
    # C_right (2x2) = A * B_right
    # Each is a 2x4x2 multiplication, which has rank 14 (known).
    # So total rank = 14 + 14 = 28.
    
    # Now I need the rank-14 decomposition for M_{2,4,2}.
    # M_{2,4,2}: A is 2x4, B is 4x2, C is 2x2.
    # This is equivalent to M_{4,2,2} by symmetry (transpose).
    # M_{4,2,2} has rank 14.
    
    # Let me construct M_{2,4,2} using Strassen on a 4x2x2.
    # Actually, let's think of A as a 2x4 matrix = [A1, A2] where A1, A2 are 2x2.
    # B is 4x2 = [[B1], [B2]] where B1, B2 are 2x2.
    # C = A1*B1 + A2*B2 (2x2 result)
    
    # This is a sum of two 2x2x2 products.
    # We can use Strassen for each, but that gives 7+7=14.
    # But wait, Strassen computes a 2x2*2x2 product with rank 7.
    # So M_{2,4,2} has rank <= 14.
    
    # Let me construct the rank-7 Strassen decomposition for 2x2x2.
    # Standard Strassen:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For M_{2,4,2} = A1*B1 + A2*B2:
    # Use Strassen for A1*B1 (rank 7) and A2*B2 (rank 7)
    # Total rank 14.
    
    # But we need to be careful about the indexing.
    # Let me construct the full 2x4x4 decomposition with rank 28.
    
    # Actually, I realize I should just construct it directly.
    # Let me define the Strassen 2x2x2 decomposition first.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition, each of shape (4, 7)."""
        # U indices: i*m + j = i*2 + j, i,j in {0,1}
        # V indices: j*p + k = j*2 + k, j,k in {0,1}
        # W indices: k*n + i = k*2 + i, k,i in {0,1}
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A00 is U[0], A11 is U[3]
        # B00 is V[0], B11 is V[3]
        # C00 gets M1, C11 gets M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # C00 = k=0,i=0 -> W[0]; C11 = k=1,i=1 -> W[3]
        
        # M2 = (A10 + A11) * B00
        # A10 is U[2], A11 is U[3], B00 is V[0]
        # C10 gets M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # C10 = k=0,i=1 -> W[2]
        
        # M3 = A00 * (B01 - B11)
        # A00 is U[0], B01 is V[1], B11 is V[3]
        # C01 gets M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # C01 = k=1,i=0 -> W[1]
        
        # M4 = A11 * (B10 - B00)
        # A11 is U[3], B10 is V[2], B00 is V[0]
        # C00 gets M4, C10 gets M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00 + A01) * B11
        # A00 is U[0], A01 is U[1], B11 is V[3]
        # C00 gets -M5, C01 gets M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # A10 is U[2], A00 is U[0], B00 is V[0], B01 is V[1]
        # C11 gets M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C11 = k=1,i=1 -> W[3]
        
        # M7 = (A01 - A11) * (B10 + B11)
        # A01 is U[1], A11 is U[3], B10 is V[2], B11 is V[3]
        # C00 gets M7, C11 gets M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    # Now for M_{2,4,4}:
    # A is 2x4, B is 4x4, C is 2x4
    # Split A = [A1, A2] (2x2 each)
    # Split B = [[B1, B2], [B3, B4]] (2x2 each)
    # C1 = A1*B1 + A2*B3 (2x2)
    # C2 = A1*B2 + A2*B4 (2x2)
    
    # For C1 = A1*B1 + A2*B3:
    # Use Strassen for A1*B1 (rank 7) and A2*B3 (rank 7)
    # For C2 = A1*B2 + A2*B4:
    # Use Strassen for A1*B2 (rank 7) and A2*B4 (rank 7)
    # Total rank = 28
    
    # Let me construct this.
    U_full = np.zeros((8, 28))  # n*m = 2*4 = 8
    V_full = np.zeros((16, 28))  # m*p = 4*4 = 16
    W_full = np.zeros((8, 28))  # p*n = 4*2 = 8
    
    # A indices: i in {0,1}, j in {0,1,2,3}
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B indices: j in {0,1,2,3}, k in {0,1,2,3}
    # B1 = B[0:2, 0:2], B2 = B[0:2, 2:4]
    # B3 = B[2:4, 0:2], B4 = B[2:4, 2:4]
    # C indices: i in {0,1}, k in {0,1,2,3}
    # C1 = C[:, 0:2], C2 = C[:, 2:4]
    
    # For C1 = A1*B1 + A2*B3:
    # Strassen for A1*B1:
    # A1 indices: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,2,3
    # B1 indices: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,2,3
    # C1[0,0] -> W[0], C1[0,1] -> W[1], C1[1,0] -> W[2], C1[1,1] -> W[3]
    
    U1, V1, W1 = strassen_2x2x2()
    # Add to columns 0-6
    U_full[0:4, 0:7] = U1
    V_full[0:4, 0:7] = V1
    W_full[0:4, 0:7] = W1
    
    # Strassen for A2*B3:
    # A2 indices: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 4,5,6,7
    # B3 indices: B[2,0], B[2,1], B[3,0], B[3,1] -> V indices 8,9,10,11
    # C1[0,0] -> W[0], C1[0,1] -> W[1], C1[1,0] -> W[2], C1[1,1] -> W[3]
    U2, V2, W2 = strassen_2x2x2()
    U_full[4:8, 7:14] = U2
    V_full[8:12, 7:14] = V2
    W_full[0:4, 7:14] = W2
    
    # For C2 = A1*B2 + A2*B4:
    # Strassen for A1*B2:
    # A1 indices: U[0:4]
    # B2 indices: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices 4,5,6,7
    # C2[0,0] -> W[4], C2[0,1] -> W[5], C2[1,0] -> W[6], C2[1,1] -> W[7]
    U3, V3, W3 = strassen_2x2x2()
    U_full[0:4, 14:21] = U3
    V_full[4:8, 14:21] = V3
    W_full[4:8, 14:21] = W3
    
    # Strassen for A2*B4:
    # A2 indices: U[4:8]
    # B4 indices: B[2,2], B[2,3], B[3,2], B[3,3] -> V indices 12,13,14,15
    # C2[0,0] -> W[4], C2[0,1] -> W[5], C2[1,0] -> W[6], C2[1,1] -> W[7]
    U4, V4, W4 = strassen_2x2x2()
    U_full[4:8, 21:28] = U4
    V_full[12:16, 21:28] = V4
    W_full[4:8, 21:28] = W4
    
    # Verify the decomposition
    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # This is the tensor T[i,j,k] which should be 1 for all valid i,j,k
    
    # Let's verify
    T_reconstructed = np.zeros((2, 4, 4))
    for i in range(2):
        for j in range(4):
            for k in range(4):
                val = 0
                for r in range(28):
                    val += U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*2+i, r]
                T_reconstructed[i, j, k] = val
    
    # The true tensor is all 1s
    if not np.allclose(T_reconstructed, 1.0):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Convert to lists of floats (or keep as integers if possible)
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
