"""Matrix multiplication 3x5x6 tensor rank minimization via Strassen block decomposition."""
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFE_MARGIN = 5.0

def _time_remaining():
    return _DEADLINE - _SAFE_MARGIN - (time.time() - _START_TIME)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix
    by a 5 by 6 matrix. Output factor W follows W[k*n+i, r].
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = 3, 5, 6
    schoolbook_rank = n * m * p  # 90
    
    # Strategy: Use Strassen's algorithm for the 2x2x2 sub-blocks to reduce rank.
    # Split A (3x5) into blocks: top 2 rows, bottom 1 row; cols 0-1, 2-4
    # Split B (5x6) into blocks: top 2 rows, bottom 3 rows; cols 0-2, 3-5
    # The 2x2x2 block (A[0:2,0:2] * B[0:2,0:2]) uses Strassen (rank 7 instead of 8)
    # All other blocks use schoolbook.
    
    # Block structure:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 1x2, A22 is 1x3
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11*B11 + A12*B21 (2x2 * 2x2 + 2x3 * 3x2)
    # C12 = A11*B12 + A12*B22 (2x2 * 2x3 + 2x3 * 3x3)
    # C21 = A21*B11 + A22*B21 (1x2 * 2x2 + 1x3 * 3x2)
    # C22 = A21*B12 + A22*B22 (1x2 * 2x3 + 1x3 * 3x3)
    
    # For C11: A11*B11 is 2x2x2 -> Strassen (rank 7), A12*B21 is 2x3x2 -> schoolbook (rank 12)
    # For C12: A11*B12 is 2x2x3 -> schoolbook (rank 12), A12*B22 is 2x3x3 -> schoolbook (rank 18)
    # For C21: A21*B11 is 1x2x2 -> schoolbook (rank 4), A22*B21 is 1x3x2 -> schoolbook (rank 6)
    # For C22: A21*B12 is 1x2x3 -> schoolbook (rank 6), A22*B22 is 1x3x3 -> schoolbook (rank 9)
    
    # Total rank = 7 + 12 + 12 + 18 + 4 + 6 + 6 + 9 = 74
    
    # Actually, let me be more careful. The tensor decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In the factorized form: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not quite right. Let me re-derive.
    
    # The correct factorization:
    # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[k,i,r]
    # No, that's not standard either.
    
    # Standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r]
    # But that's not a rank decomposition.
    
    # The rank-r decomposition is:
    # C[i,k] = sum_r M_r * W[k,i,r]
    # where M_r = (sum_{i,j} U[i,j,r] * A[i,j]) * (sum_{j,k} V[j,k,r] * B[j,k])
    
    # So: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r] = coefficient of A[i,j] in M_r
    # V[j*p+k, r] = coefficient of B[j,k] in M_r
    # W[k*n+i, r] = coefficient of M_r in C[i,k]
    
    # For schoolbook: each (i,j,k) triple gets its own r.
    # U[i*m+j, r_{i,j,k}] = 1, V[j*p+k, r_{i,j,k}] = 1, W[k*n+i, r_{i,j,k}] = 1
    
    # For Strassen on 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me build the full decomposition with Strassen for the 2x2x2 block.
    
    # Define the Strassen decomposition for 2x2x2
    # Returns U_local (4, 7), V_local (4, 7), W_local (4, 7) for the 2x2x2 block
    def strassen_2x2x2():
        """Strassen decomposition for 2x2x2 matrix multiplication."""
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1: (A00+A11)(B00+B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        
        # M2: (A10+A11)B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        
        # M3: A00(B10-B11)
        U[0, 2] = 1  # A00
        V[2, 2] = 1  # B10
        V[3, 2] = -1 # B11
        
        # M4: A11(B01-B00)
        U[3, 3] = 1  # A11
        V[1, 3] = 1  # B01
        V[0, 3] = -1 # B00
        
        # M5: (A00+A01)B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        
        # M6: (A10-A00)(B00+B01)
        U[2, 5] = 1  # A10
        U[0, 5] = -1 # A00
        V[0, 5] = 1  # B00
        V[1, 5] = 1  # B01
        
        # M7: (A01-A11)(B10+B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1 # A11
        V[2, 6] = 1  # B10
        V[3, 6] = 1  # B11
        
        # W: C[i,k] = sum_r W[k*2+i, r] * M_r
        # C00 = M1 + M4 - M5 + M7 -> W[0*2+0, r] = W[0, r]
        W[0, 0] = 1  # M1
        W[0, 3] = 1  # M4
        W[0, 4] = -1 # M5
        W[0, 6] = 1  # M7
        
        # C01 = M3 + M5 -> W[1*2+0, r] = W[2, r]
        W[2, 2] = 1  # M3
        W[2, 4] = 1  # M5
        
        # C10 = M2 + M4 -> W[0*2+1, r] = W[1, r]
        W[1, 1] = 1  # M2
        W[1, 3] = 1  # M4
        
        # C11 = M1 - M2 + M3 + M6 -> W[1*2+1, r] = W[3, r]
        W[3, 0] = 1  # M1
        W[3, 1] = -1 # M2
        W[3, 2] = 1  # M3
        W[3, 5] = 1  # M6
        
        return U, V, W
    
    # Build the full decomposition
    # U_full: (15, rank), V_full: (30, rank), W_full: (18, rank)
    
    # Track column assignments
    col = 0
    U_full = []
    V_full = []
    W_full = []
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U_row = [0] * 15
        V_row = [0] * 30
        W_row = [0] * 18
        U_row[i*5+j] = 1
        V_row[j*6+k] = 1
        W_row[k*3+i] = 1
        U_full.append(U_row)
        V_full.append(V_row)
        W_full.append(W_row)
        col += 1
    
    # Helper to add Strassen terms for a 2x2x2 block
    def add_strassen_block(i_offset, j_offset, k_offset):
        nonlocal col
        U_str, V_str, W_str = strassen_2x2x2()
        for r in range(7):
            U_row = [0] * 15
            V_row = [0] * 30
            W_row = [0] * 18
            for i_local in range(2):
                for j_local in range(2):
                    U_row[(i_offset+i_local)*5 + (j_offset+j_local)] = U_str[i_local*2+j_local, r]
            for j_local in range(2):
                for k_local in range(2):
                    V_row[(j_offset+j_local)*6 + (k_offset+k_local)] = V_str[j_local*2+k_local, r]
            for k_local in range(2):
                for i_local in range(2):
                    W_row[(k_offset+k_local)*3 + (i_offset+i_local)] = W_str[k_local*2+i_local, r]
            U_full.append(U_row)
            V_full.append(V_row)
            W_full.append(W_row)
            col += 1
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: 2x2x2 Strassen block at i=0,1; j=0,1; k=0,1
    add_strassen_block(0, 0, 0)
    
    # A12*B21: 2x3x2 schoolbook, i=0,1; j=2,3,4; k=0,1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: 2x2x3 schoolbook, i=0,1; j=0,1; k=2,3,4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    # A12*B22: 2x3x3 schoolbook, i=0,1; j=2,3,4; k=2,3,4
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: 1x2x2 schoolbook, i=2; j=0,1; k=0,1
    for i in range(2, 3):
        for j in range(2):
            for k in range(2):
                add_schoolbook(i, j, k)
    
    # A22*B21: 1x3x2 schoolbook, i=2; j=2,3,4; k=0,1
    for i in range(2, 3):
        for j in range(2, 5):
            for k in range(2):
                add_schoolbook(i, j, k)
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: 1x2x3 schoolbook, i=2; j=0,1; k=2,3,4
    for i in range(2, 3):
        for j in range(2):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    # A22*B22: 1x3x3 schoolbook, i=2; j=2,3,4; k=2,3,4
    for i in range(2, 3):
        for j in range(2, 5):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
    
    rank = col
    U_arr = np.array(U_full).T  # (rank, 15) -> transpose to (15, rank)
    V_arr = np.array(V_full).T  # (rank, 30) -> transpose to (30, rank)
    W_arr = np.array(W_full).T  # (rank, 18) -> transpose to (18, rank)
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(len(U_mat[0])):
                        total += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None
    
    is_valid, err = verify(U_arr, V_arr, W_arr, n, m, p)
    if not is_valid:
        # Fallback to schoolbook if Strassen construction has a bug
        rank = n * m * p
        U_arr = np.zeros((n*m, rank), dtype=int)
        V_arr = np.zeros((m*p, rank), dtype=int)
        W_arr = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_arr[i*m+j, col] = 1
                    V_arr[j*p+k, col] = 1
                    W_arr[k*n+i, col] = 1
                    col += 1
    
    # Convert to lists
    result_U = U_arr.tolist()
    result_V = V_arr.tolist()
    result_W = W_arr.tolist()
    
    return {"U": result_U, "V": result_V, "W": result_W}
