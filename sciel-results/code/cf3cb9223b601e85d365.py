"""Rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def _strassen_2x2_coeffs():
    """
    Return (U, V, W) for Strassen 2x2 matrix multiplication, rank 7.
    Convention:
      U has shape (4, 7): U[i*2+j, c] is coefficient of A[i,j] in the c-th bilinear form.
      V has shape (4, 7): V[j*2+k, c] is coefficient of B[j,k] in the c-th bilinear form.
      W has shape (4, 7): W[k*2+i, c] is coefficient of C[i,k] in the reconstruction.
    
    Strassen's 7 products (0-indexed for i,j,k):
      M1 = (a00+a11)(b00+b11)
      M2 = (a10+a11) b00
      M3 = a00 (b01 - b11)
      M4 = a11 (b10 - b00)
      M5 = (a00+a01) b11
      M6 = (a10 - a00) (b00+b10)
      M7 = (a01 - a11) (b10+b11)
      
    Reconstruction:
      c00 = M1 + M4 - M5 + M7
      c01 = M3 + M5
      c10 = M2 + M4
      c11 = M1 - M2 + M3 + M7
    """
    # U[i*2+j, c]
    U = np.zeros((4, 7))
    # V[j*2+k, c]
    V = np.zeros((4, 7))
    # W[k*2+i, c]
    W = np.zeros((4, 7))
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1  # a00, a11
    V[0, 0] = 1; V[3, 0] = 1  # b00, b11
    W[0, 0] = 1; W[3, 0] = 1  # c00, c11
    
    # M2 = (a10+a11) b00
    U[2, 1] = 1; U[3, 1] = 1  # a10, a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # c10
    
    # M3 = a00 (b01 - b11)
    U[0, 2] = 1  # a00
    V[1, 2] = 1; V[3, 2] = -1  # b01 - b11
    W[1, 2] = 1  # c01
    
    # M4 = a11 (b10 - b00)
    U[3, 3] = 1  # a11
    V[2, 3] = 1; V[0, 3] = -1  # b10 - b00
    W[0, 3] = 1; W[2, 3] = 1  # c00, c10
    
    # M5 = (a00+a01) b11
    U[0, 4] = 1; U[1, 4] = 1  # a00, a01
    V[3, 4] = 1  # b11
    W[0, 4] = -1; W[1, 4] = 1  # c00, c01
    
    # M6 = (a10 - a00) (b00+b10)
    U[2, 5] = 1; U[0, 5] = -1  # a10 - a00
    V[0, 5] = 1; V[2, 5] = 1  # b00 + b10
    # M6 is not used in standard reconstruction? Let me check.
    # Actually M6 is used in some formulations. Let me re-verify.
    # Standard Strassen:
    # p1 = (A11 + A22)(B11 + B22)
    # p2 = (A21 + A22)B11
    # p3 = A11(B21 - B22)
    # p4 = A22(B12 - B11)
    # p5 = (A11 + A12)B22
    # p6 = (A21 - A11)(B11 + B12)
    # p7 = (A12 - A22)(B21 + B22)
    # C11 = p1 - p4 + p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    # 
    # Wait, C22 uses p6! Let me re-check.
    # Yes, C22 = p1 - p2 + p3 + p6 in some formulations.
    # Let me use the standard formulation carefully.
    
    # Let me restart with the correct standard Strassen.
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # c10
    
    # p3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # c01
    W[3, 2] = 1  # c11
    
    # p4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1  # c00
    W[2, 3] = 1   # c10
    
    # p5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1  # c00
    W[1, 4] = 1  # c01
    
    # p6 = (a10-a00)(b00+b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # c11
    
    # p7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # c00
    W[3, 6] = 1  # c11
    
    # Verify reconstruction:
    # c00 = p1 - p4 + p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # Check W columns:
    # W[k*2+i, c] where c is the product index
    # c00 (i=0,k=0): index 0
    # c01 (i=0,k=1): index 1
    # c10 (i=1,k=0): index 2
    # c11 (i=1,k=1): index 3
    
    # p1: c00=1, c11=1 -> W[0,0]=1, W[3,0]=1 ✓
    # p2: c10=1 -> W[2,1]=1 ✓
    # p3: c01=1, c11=1 -> W[1,2]=1, W[3,2]=1 ✓
    # p4: c00=-1, c10=1 -> W[0,3]=-1, W[2,3]=1 ✓
    # p5: c00=1, c01=1 -> W[0,4]=1, W[1,4]=1 ✓
    # p6: c11=1 -> W[3,5]=1 ✓
    # p7: c00=1, c11=1 -> W[0,6]=1, W[3,6]=1 ✓
    
    return U, V, W

def construct():
    """
    Rank-14 decomposition of 2x2 * 2x4 matrix multiplication.
    The output C is 2x4. We can view this as two 2x2 matrix multiplications:
      C[:, 0:2] = A * B[:, 0:2]
      C[:, 2:4] = A * B[:, 2:4]
    
    Each 2x2 multiplication can be done with Strassen (rank 7), so total rank = 14.
    
    Indexing:
      A is 2x2: A[i, j] for i,j in {0,1}
      B is 2x4: B[j, k] for j in {0,1}, k in {0,1,2,3}
      C is 2x4: C[i, k] for i in {0,1}, k in {0,1,2,3}
    
    U has shape (4, 14): U[i*2+j, c]
    V has shape (8, 14): V[j*4+k, c]
    W has shape (8, 14): W[k*2+i, c]  (trace convention: W[k*n+i, r] where n=2)
    """
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin before 120s limit
    
    # Get Strassen coefficients for one 2x2 block
    U_s, V_s, W_s = _strassen_2x2_coeffs()
    
    # For the full 2x2 * 2x4 multiplication:
    # Columns 0-6: first 2x2 block (B columns 0,1)
    # Columns 7-13: second 2x2 block (B columns 2,3)
    
    rank = 14
    n, m, p = 2, 2, 4
    
    U = np.zeros((n * m, rank))  # (4, 14)
    V = np.zeros((m * p, rank))  # (8, 14)
    W = np.zeros((p * n, rank))  # (8, 14)
    
    # First block: B columns 0,1
    # U is same for both blocks (A doesn't change)
    U[:, 0:7] = U_s
    
    # V for first block: V[j*4+k, c] where k in {0,1}
    # V_s[j*2+k, c] where k in {0,1} maps to V[j*4+k, c]
    for c in range(7):
        for j in range(2):
            for k in range(2):
                V[j * 4 + k, c] = V_s[j * 2 + k, c]
    
    # W for first block: W[k*2+i, c] where k in {0,1}
    # This is already in the right form since W_s[k*2+i, c]
    W[0:4, 0:7] = W_s
    
    # Second block: B columns 2,3
    U[:, 7:14] = U_s
    
    # V for second block: V[j*4+k, c+7] where k in {2,3}
    # Map V_s[j*2+k', c] where k' in {0,1} to V[j*4+(k'+2), c+7]
    for c in range(7):
        for j in range(2):
            for k_prime in range(2):
                V[j * 4 + (k_prime + 2), c + 7] = V_s[j * 2 + k_prime, c]
    
    # W for second block: W[k*2+i, c+7] where k in {2,3}
    # W_s[k'*2+i, c] where k' in {0,1} maps to W[(k'+2)*2+i, c+7]
    for c in range(7):
        for k_prime in range(2):
            for i in range(2):
                W[(k_prime + 2) * 2 + i, c + 7] = W_s[k_prime * 2 + i, c]
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Clean up floating point artifacts (should be integers)
    U_list = [[int(round(x)) for x in row] for row in U_list]
    V_list = [[int(round(x)) for x in row] for row in V_list]
    W_list = [[int(round(x)) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
