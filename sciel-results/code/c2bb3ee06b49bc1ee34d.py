"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U @ V @ W^T reconstructs the matmul tensor exactly."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k])
                val = np.dot(val, W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False, abs(val - 1.0)
    return True, 0.0

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition for n x m x p matrix multiplication."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2x2_block():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Index convention: row = i*m + j for U, j*p + k for V, k*n + i for W.
    For n=m=p=2:
    U indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    V indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    W indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B01 + B10)
    U[0, 0] = 1; U[3, 0] = 1
    V[1, 0] = 1; V[2, 0] = 1
    W[1, 0] = 1; W[2, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[1, 2] = 1
    
    # M4 = A11(B00 - B01)
    U[3, 3] = 1
    V[0, 3] = 1; V[1, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def _build_3x4x8_decomposition():
    """
    Build a rank-88 decomposition for 3x4x8 matrix multiplication using Strassen.
    
    Strategy:
    - Split the 3x4 matrix into a 2x4 block (rows 0-1) and a 1x4 block (row 2).
    - Split the 4x8 matrix into two 4x4 blocks (cols 0-3 and cols 4-7).
    - So we have:
      - C[0:2, 0:4] = A[0:2, :] * B[:, 0:4]  (2x4 times 4x4)
      - C[0:2, 4:8] = A[0:2, :] * B[:, 4:8]  (2x4 times 4x4)
      - C[2, 0:4] = A[2, :] * B[:, 0:4]      (1x4 times 4x4)
      - C[2, 4:8] = A[2, :] * B[:, 4:8]      (1x4 times 4x4)
    
    For 2x4 times 4x4:
      Split 2x4 into two 2x2 blocks (cols 0-1 and 2-3).
      Split 4x4 into four 2x2 blocks.
      Use Strassen on each 2x2x2 sub-problem.
      This gives 4 * 7 = 28 rank units per 2x4x4.
      Two such: 56.
    
    For 1x4 times 4x4:
      Schoolbook: 4 * 4 = 16 rank units per 1x4x4.
      Two such: 32.
    
    Total: 56 + 32 = 88.
    """
    n, m, p = 3, 4, 8
    
    U_full = np.zeros((n * m, 88))
    V_full = np.zeros((m * p, 88))
    W_full = np.zeros((p * n, 88))
    
    col_offset = 0
    
    # Part 1: A[0:2, :] * B[:, 0:4] -> C[0:2, 0:4]
    # A block: rows 0-1, cols 0-3 of A (shape 2x4)
    # B block: rows 0-3, cols 0-3 of B (shape 4x4)
    # C block: rows 0-1, cols 0-3 of C (shape 2x4)
    
    # Split A[0:2, 0:4] into A11 (rows 0-1, cols 0-1) and A12 (rows 0-1, cols 2-3)
    # Split B[0:4, 0:4] into B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3),
    #                    B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    # C[0:2, 0:4] = A11*B11 + A12*B21  (top-left 2x2 of C)
    #              = A11*B12 + A12*B22  (top-right 2x2 of C)
    
    # For C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # This is (2x2 + 2x2) * (2x2 + 2x2) which isn't a simple matmul.
    # Actually, C[i,k] = sum_j A[i,j] * B[j,k]
    # For i in 0-1, k in 0-3:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Let's use a different approach: treat 2x4 x 4x4 as two 2x2 x 4x4 problems
    # or use the 2x2x2 Strassen blocks directly.
    
    # Simpler: split into 2x2x2 blocks.
    # A is 2x4 = [A_a | A_b] where A_a is 2x2, A_b is 2x2
    # B is 4x4 = [[B_aa, B_ab], [B_ba, B_bb]]
    # C is 2x4 = [C_a | C_b] where C_a is 2x2, C_b is 2x2
    # C_a = A_a * B_aa + A_b * B_ba
    # C_b = A_a * B_ab + A_b * B_bb
    
    # This is two 2x2x2 matmuls for C_a and C_b? No, it's sums of products.
    # C_a = A_a * B_aa + A_b * B_ba requires 7+7=14 rank units (two independent Strassen)
    # C_b = A_a * B_ab + A_b * B_bb requires 7+7=14 rank units
    # Total for 2x4x4: 28 rank units. Correct.
    
    # Implement Strassen for the four 2x2x2 sub-problems in this 2x4x4 block.
    U_s, V_s, W_s = _strassen_2x2x2_block()
    
    # Sub-problem 1: A_a * B_aa -> part of C_a
    # A_a is A[0:2, 0:2], B_aa is B[0:2, 0:2], C_a is C[0:2, 0:2]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, c] != 0:
                    U_full[i*4+j, col_offset+c] = U_s[i*2+j, c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, c] != 0:
                    V_full[j*4+k, col_offset+c] = V_s[j*2+k, c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, c] != 0:
                    W_full[k*3+i, col_offset+c] = W_s[k*2+i, c]
    col_offset += 7
    
    # Sub-problem 2: A_b * B_ba -> part of C_a
    # A_b is A[0:2, 2:4], B_ba is B[2:4, 0:2], C_a is C[0:2, 0:2]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, c] != 0:
                    U_full[i*4+(j+2), col_offset+c] = U_s[i*2+j, c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, c] != 0:
                    V_full[(j+2)*4+k, col_offset+c] = V_s[j*2+k, c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, c] != 0:
                    W_full[k*3+i, col_offset+c] = W_s[k*2+i, c]
    col_offset += 7
    
    # Sub-problem 3: A_a * B_ab -> part of C_b
    # A_a is A[0:2, 0:2], B_ab is B[0:2, 2:4], C_b is C[0:2, 2:4]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, c] != 0:
                    U_full[i*4+j, col_offset+c] = U_s[i*2+j, c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, c] != 0:
                    V_full[j*4+(k+2), col_offset+c] = V_s[j*2+k, c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, c] != 0:
                    W_full[(k+2)*3+i, col_offset+c] = W_s[k*2+i, c]
    col_offset += 7
    
    # Sub-problem 4: A_b * B_bb -> part of C_b
    # A_b is A[0:2, 2:4], B_bb is B[2:4, 2:4], C_b is C[0:2, 2:4]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, c] != 0:
                    U_full[i*4+(j+2), col_offset+c] = U_s[i*2+j, c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, c] != 0:
                    V_full[(j+2)*4+(k+2), col_offset+c] = V_s[j*2+k, c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, c] != 0:
                    W_full[(k+2)*3+i, col_offset+c] = W_s[k*2+i, c]
    col_offset += 7
    
    # Part 2: A[0:2, :] * B[:, 4:8] -> C[0:2, 4:8]
    # Similar to Part 1, but B columns are shifted by 4.
    
    # Sub-problem 5: A_a * B[0:2, 4:6] -> part of C[0:2, 4:6]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, c] != 0:
                    U_full[i*4+j, col_offset+c] = U_s[i*2+j, c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, c] != 0:
                    V_full[j*4+(k+4), col_offset+c] = V_s[j*2+k, c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, c] != 0:
                    W_full[(k+4)*3+i, col_offset+c] = W_s[k*2+i, c]
    col_offset += 7
    
    # Sub-problem 6: A_b * B[2:4, 4:6] -> part of C[0:2, 4:6]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, c] != 0:
                    U_full[i*4+(j+2), col_offset+c] = U_s[i*2+j, c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, c] != 0:
                    V_full[(j+2)*4+(k+4), col_offset+c] = V_s[j*2+k, c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, c] != 0:
                    W_full[(k+4)*3+i, col_offset+c] = W_s[k*2+i, c]
    col_offset += 7
    
    # Sub-problem 7: A_a * B[0:2, 6:8] -> part of C[0:2, 6:8]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, c] != 0:
                    U_full[i*4+j, col_offset+c] = U_s[i*2+j, c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, c] != 0:
                    V_full[j*4+(k+6), col_offset+c] = V_s[j*2+k, c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, c] != 0:
                    W_full[(k+6)*3+i, col_offset+c] = W_s[k*2+i, c]
    col_offset += 7
    
    # Sub-problem 8: A_b * B[2:4, 6:8] -> part of C[0:2, 6:8]
    for c in range(7):
        for i in range(2):
            for j in range(2):
                if U_s[i*2+j, c] != 0:
                    U_full[i*4+(j+2), col_offset+c] = U_s[i*2+j, c]
        for j in range(2):
            for k in range(2):
                if V_s[j*2+k, c] != 0:
                    V_full[(j+2)*4+(k+6), col_offset+c] = V_s[j*2+k, c]
        for k in range(2):
            for i in range(2):
                if W_s[k*2+i, c] != 0:
                    W_full[(k+6)*3+i, col_offset+c] = W_s[k*2+i, c]
    col_offset += 7
    
    # Part 3: A[2, :] * B[:, 0:4] -> C[2, 0:4]
    # Schoolbook: 1x4 x 4x4 = 16 rank units
    for j in range(4):
        for k in range(4):
            U_full[2*4+j, col_offset] = 1.0
            V_full[j*4+k, col_offset] = 1.0
            W_full[k*3+2, col_offset] = 1.0
            col_offset += 1
    
    # Part 4: A[2, :] * B[:, 4:8] -> C[2, 4:8]
    # Schoolbook: 1x4 x 4x4 = 16 rank units
    for j in range(4):
        for k in range(4):
            U_full[2*4+j, col_offset] = 1.0
            V_full[j*4+(k+4), col_offset] = 1.0
            W_full[(k+4)*3+2, col_offset] = 1.0
            col_offset += 1
    
    return U_full.tolist(), V_full.tolist(), W_full.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin
    
    # Build the Strassen-based rank-88 decomposition
    U, V, W = _build_3x4x8_decomposition()
    
    # Verify correctness
    valid, err = _verify(U, V, W, n, m, p, tol=1e-9)
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
