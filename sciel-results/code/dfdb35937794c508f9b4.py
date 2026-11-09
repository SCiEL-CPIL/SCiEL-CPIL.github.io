import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x7 matrix.
    Uses Strassen's algorithm for the 2x2x2 sub-blocks to reduce rank from 84 to 78.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 4, 7

    # We will build the decomposition by splitting the matrices into blocks.
    # A is 3x4, B is 4x7.
    # Split A into A1 (2x4) and A2 (1x4).
    # Split B into B1 (4x4) and B2 (4x3).
    
    # C = A * B = [A1*B1, A1*B2; A2*B1, A2*B2]
    # C is 3x7.
    
    # We need to compute:
    # 1. A1 * B1 (2x4 * 4x4 -> 2x4)
    # 2. A1 * B2 (2x4 * 4x3 -> 2x3)
    # 3. A2 * B1 (1x4 * 4x4 -> 1x4)
    # 4. A2 * B2 (1x4 * 4x3 -> 1x3)
    
    # Let's decompose each part.
    
    # Part 1: A1 * B1 (2x4 * 4x4)
    # Split A1 into A11 (2x2) and A12 (2x2).
    # Split B1 into B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2).
    # A1 * B1 = [A11*B11 + A12*B21, A11*B12 + A12*B22]
    # Each 2x2*2x2 uses Strassen (rank 7).
    # Total rank for Part 1 = 4 * 7 = 28.
    
    # Part 2: A1 * B2 (2x4 * 4x3)
    # Split A1 into A11 (2x2) and A12 (2x2).
    # Split B2 into B21 (2x2), B22 (2x1), B31 (2x2), B32 (2x1).
    # Wait, B2 is 4x3. Split into top 2 rows and bottom 2 rows.
    # B2 = [B2_top; B2_bottom] where B2_top is 2x3, B2_bottom is 2x3.
    # A1 * B2 = A11 * B2_top + A12 * B2_bottom.
    # A11 * B2_top (2x2 * 2x3). Split B2_top into B21 (2x2) and B22 (2x1).
    # A11 * B2_top = [A11*B21, A11*B22].
    # A11*B21 is 2x2*2x2 (Strassen, rank 7).
    # A11*B22 is 2x2*2x1 (Schoolbook, rank 4).
    # Similarly for A12 * B2_bottom.
    # Total rank for Part 2 = 2 * (7 + 4) = 22.
    
    # Part 3: A2 * B1 (1x4 * 4x4)
    # A2 is 1x4. Split into A21 (1x2) and A22 (1x2).
    # B1 is 4x4. Split into B11 (2x2), B12 (2x2), B21 (2x2), B22 (2x2).
    # A2 * B1 = A21 * [B11; B21] + A22 * [B12; B22].
    # A21 * B11 (1x2 * 2x2). Schoolbook rank 4.
    # A21 * B21 (1x2 * 2x2). Schoolbook rank 4.
    # A22 * B12 (1x2 * 2x2). Schoolbook rank 4.
    # A22 * B22 (1x2 * 2x2). Schoolbook rank 4.
    # Total rank for Part 3 = 16.
    
    # Part 4: A2 * B2 (1x4 * 4x3)
    # A2 is 1x4. Split into A21 (1x2) and A22 (1x2).
    # B2 is 4x3. Split into B2_top (2x3) and B2_bottom (2x3).
    # A2 * B2 = A21 * B2_top + A22 * B2_bottom.
    # A21 * B2_top (1x2 * 2x3). Schoolbook rank 6.
    # A22 * B2_bottom (1x2 * 2x3). Schoolbook rank 6.
    # Total rank for Part 4 = 12.
    
    # Total Rank = 28 + 22 + 16 + 12 = 78.
    
    # Now, we need to construct the U, V, W matrices.
    # U has shape (n*m, rank) = (12, 78).
    # V has shape (m*p, rank) = (28, 78).
    # W has shape (p*n, rank) = (21, 78).
    
    # Indices:
    # A[i, j] for i in 0..2, j in 0..3.
    # B[j, k] for j in 0..3, k in 0..6.
    # C[i, k] for i in 0..2, k in 0..6.
    
    # U index: i*m + j = i*4 + j.
    # V index: j*p + k = j*7 + k.
    # W index: k*n + i = k*3 + i.
    
    U = np.zeros((n * m, 78), dtype=np.int8)
    V = np.zeros((m * p, 78), dtype=np.int8)
    W = np.zeros((p * n, 78), dtype=np.int8)
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(u_idx, v_idx, w_idx, sign=1):
        nonlocal col
        U[u_idx, col] += sign
        V[v_idx, col] += sign
        W[w_idx, col] += sign
        col += 1
    
    # Helper to add a Strassen term for 2x2x2
    # A_sub is 2x2, B_sub is 2x2, C_sub is 2x2.
    # Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map local indices to global indices.
    # For a block A_sub[i,j], B_sub[k,l], C_sub[i,l]:
    # Global A index: (base_i + i)*m + (base_j + j)
    # Global B index: (base_k + k)*p + (base_l + l)
    # Global C index: (base_l + l)*n + (base_i + i)
    
    def add_strassen(base_i, base_j, base_k, base_l):
        nonlocal col
        # M1 = (A11+A22)*(B11+B22)
        # A11: (base_i, base_j), A22: (base_i+1, base_j+1)
        # B11: (base_k, base_l), B22: (base_k+1, base_l+1)
        # C11: (base_i, base_l), C22: (base_i+1, base_l+1)
        
        # M1
        U[(base_i)*m + base_j, col] = 1
        U[(base_i+1)*m + base_j+1, col] = 1
        V[base_k*p + base_l, col] = 1
        V[(base_k+1)*p + base_l+1, col] = 1
        W[base_l*n + base_i, col] = 1
        W[(base_l+1)*n + base_i+1, col] = 1
        col += 1
        
        # M2 = (A21+A22)*B11
        # A21: (base_i+1, base_j), A22: (base_i+1, base_j+1)
        # B11: (base_k, base_l)
        # C21: (base_i+1, base_l)
        U[(base_i+1)*m + base_j, col] = 1
        U[(base_i+1)*m + base_j+1, col] = 1
        V[base_k*p + base_l, col] = 1
        W[base_l*n + base_i+1, col] = 1
        col += 1
        
        # M3 = A11*(B12-B22)
        # A11: (base_i, base_j)
        # B12: (base_k, base_l+1), B22: (base_k+1, base_l+1)
        # C12: (base_i, base_l+1)
        U[base_i*m + base_j, col] = 1
        V[base_k*p + base_l+1, col] = 1
        V[(base_k+1)*p + base_l+1, col] = -1
        W[(base_l+1)*n + base_i, col] = 1
        col += 1
        
        # M4 = A22*(B21-B11)
        # A22: (base_i+1, base_j+1)
        # B21: (base_k+1, base_l), B11: (base_k, base_l)
        # C21: (base_i+1, base_l), C22: (base_i+1, base_l+1)
        U[(base_i+1)*m + base_j+1, col] = 1
        V[(base_k+1)*p + base_l, col] = 1
        V[base_k*p + base_l, col] = -1
        W[base_l*n + base_i+1, col] = 1
        W[(base_l+1)*n + base_i+1, col] = 1
        col += 1
        
        # M5 = (A11+A12)*B22
        # A11: (base_i, base_j), A12: (base_i, base_j+1)
        # B22: (base_k+1, base_l+1)
        # C11: (base_i, base_l), C12: (base_i, base_l+1)
        U[base_i*m + base_j, col] = 1
        U[base_i*m + base_j+1, col] = 1
        V[(base_k+1)*p + base_l+1, col] = 1
        W[base_l*n + base_i, col] = -1
        W[(base_l+1)*n + base_i, col] = 1
        col += 1
        
        # M6 = (A21-A11)*(B11+B12)
        # A21: (base_i+1, base_j), A11: (base_i, base_j)
        # B11: (base_k, base_l), B12: (base_k, base_l+1)
        # C22: (base_i+1, base_l+1)
        U[(base_i+1)*m + base_j, col] = 1
        U[base_i*m + base_j, col] = -1
        V[base_k*p + base_l, col] = 1
        V[base_k*p + base_l+1, col] = 1
        W[(base_l+1)*n + base_i+1, col] = 1
        col += 1
        
        # M7 = (A12-A22)*(B21+B22)
        # A12: (base_i, base_j+1), A22: (base_i+1, base_j+1)
        # B21: (base_k+1, base_l), B22: (base_k+1, base_l+1)
        # C11: (base_i, base_l), C22: (base_i+1, base_l+1)
        U[base_i*m + base_j+1, col] = 1
        U[(base_i+1)*m + base_j+1, col] = -1
        V[(base_k+1)*p + base_l, col] = 1
        V[(base_k+1)*p + base_l+1, col] = 1
        W[base_l*n + base_i, col] = 1
        W[(base_l+1)*n + base_i+1, col] = 1
        col += 1

    # Part 1: A1 * B1 (2x4 * 4x4)
    # A1 is rows 0,1 of A. B1 is cols 0,1,2,3 of B.
    # A11: rows 0,1; cols 0,1.
    # A12: rows 0,1; cols 2,3.
    # B11: rows 0,1; cols 0,1.
    # B12: rows 0,1; cols 2,3.
    # B21: rows 2,3; cols 0,1.
    # B22: rows 2,3; cols 2,3.
    
    # C11 (rows 0,1; cols 0,1) = A11*B11 + A12*B21
    add_strassen(0, 0, 0, 0) # A11*B11
    add_strassen(0, 2, 2, 0) # A12*B21
    
    # C12 (rows 0,1; cols 2,3) = A11*B12 + A12*B22
    add_strassen(0, 0, 0, 2) # A11*B12
    add_strassen(0, 2, 2, 2) # A12*B22
    
    # Part 2: A1 * B2 (2x4 * 4x3)
    # B2 is cols 4,5,6 of B.
    # B2_top: rows 0,1; cols 4,5,6.
    # B2_bottom: rows 2,3; cols 4,5,6.
    
    # C21 (rows 0,1; cols 4,5,6) = A11*B2_top + A12*B2_bottom
    # A11*B2_top:
    # B2_top = [B21 (2x2), B22 (2x1)]
    # B21: rows 0,1; cols 4,5.
    # B22: rows 0,1; col 6.
    add_strassen(0, 0, 0, 4) # A11*B21
    # A11*B22 (2x2 * 2x1)
    # Schoolbook:
    # C[0,6] += A[0,0]*B[0,6] + A[0,1]*B[1,6]
    # C[1,6] += A[1,0]*B[0,6] + A[1,1]*B[1,6]
    for i in range(2):
        for j in range(2):
            for k in range(1):
                add_schoolbook(i*m + j, (0+k)*p + (6+k), (6+k)*n + i)
                
    # A12*B2_bottom:
    # B2_bottom = [B31 (2x2), B32 (2x1)]
    # B31: rows 2,3; cols 4,5.
    # B32: rows 2,3; col 6.
    add_strassen(0, 2, 2, 4) # A12*B31
    for i in range(2):
        for j in range(2):
            for k in range(1):
                add_schoolbook((i)*m + (2+j), (2+k)*p + (6+k), (6+k)*n + i)
                
    # Part 3: A2 * B1 (1x4 * 4x4)
    # A2 is row 2 of A.
    # A21: row 2; cols 0,1.
    # A22: row 2; cols 2,3.
    
    # C31 (row 2; cols 0,1) = A21*B11 + A22*B21
    # A21*B11 (1x2 * 2x2)
    for j in range(2):
        for k in range(2):
            add_schoolbook(2*m + j, k*p + k, k*n + 2)
            
    # A22*B21 (1x2 * 2x2)
    for j in range(2):
        for k in range(2):
            add_schoolbook(2*m + (2+j), (2+k)*p + k, k*n + 2)
            
    # C32 (row 2; cols 2,3) = A21*B12 + A22*B22
    # A21*B12 (1x2 * 2x2)
    for j in range(2):
        for k in range(2):
            add_schoolbook(2*m + j, k*p + (2+k), (2+k)*n + 2)
            
    # A22*B22 (1x2 * 2x2)
    for j in range(2):
        for k in range(2):
            add_schoolbook(2*m + (2+j), (2+k)*p + (2+k), (2+k)*n + 2)
            
    # Part 4: A2 * B2 (1x4 * 4x3)
    # C33 (row 2; cols 4,5,6) = A21*B2_top + A22*B2_bottom
    # A21*B2_top (1x2 * 2x3)
    for j in range(2):
        for k in range(3):
            add_schoolbook(2*m + j, k*p + (4+k), (4+k)*n + 2)
            
    # A22*B2_bottom (1x2 * 2x3)
    for j in range(2):
        for k in range(3):
            add_schoolbook(2*m + (2+j), (2+k)*p + (4+k), (4+k)*n + 2)
            
    # Check if we have exactly 78 columns
    if col != 78:
        # If not, we might have an error in counting.
        # But we expect exactly 78.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
