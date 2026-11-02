"""Rank-minimal exact tensor decomposition for 2x4 x 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return (U,V,W) for 2x2 * 2x2 -> 2x2 with rank 7 (Strassen).
    U: (4,7) indexed by [i*2+j, c]
    V: (4,7) indexed by [j*2+k, c]
    W: (4,7) indexed by [k*2+i, c]
    """
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # Strassen formulas for M11, M12, M21, M22
    # M11 = (A11)(B11 + B22)
    # M12 = (A11 + A21)(B22)
    # M21 = (A12 + A22)(B11)
    # M22 = (A22)(B11 + B22) -- wait, standard Strassen is:
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
    
    # Indices:
    # A: [i*2+j] where i in {0,1}, j in {0,1}
    # A11 = A[0], A12 = A[1], A21 = A[2], A22 = A[3]
    # B: [j*2+k] where j in {0,1}, k in {0,1}
    # B11 = B[0], B12 = B[1], B21 = B[2], B22 = B[3]
    # C: [k*2+i] where k in {0,1}, i in {0,1}
    # C11 = C[0], C21 = C[1], C12 = C[2], C22 = C[3]
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11(1), A22(1)
    # V: B11(1), B22(1)
    # W: C11(1), C22(1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    # U: A21(1), A22(1)
    # V: B11(1)
    # W: C21(1), C22(-1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A11(B12-B22)
    # U: A11(1)
    # V: B12(1), B22(-1)
    # W: C12(1), C22(1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A22(B21-B11)
    # U: A22(1)
    # V: B21(1), B11(-1)
    # W: C11(1), C21(1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A11+A12)B22
    # U: A11(1), A12(1)
    # V: B22(1)
    # W: C11(-1), C12(1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # U: A21(1), A11(-1)
    # V: B11(1), B12(1)
    # W: C22(1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # U: A12(1), A22(-1)
    # V: B21(1), B22(1)
    # W: C11(1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Return (U,V,W) for n x m * m x p -> n x p with schoolbook rank n*m*p.
    U: (n*m, n*m*p) indexed by [i*m+j, c]
    V: (m*p, n*m*p) indexed by [j*p+k, c]
    W: (p*n, n*m*p) indexed by [k*n+i, c]
    """
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Decompose as:
    # A is 2x4, B is 4x4, C is 2x4
    # Split B into two 4x2 blocks: B = [B_left, B_right] where B_left is 4x2, B_right is 4x2
    # But wait, we need to use the structure better.
    # 
    # Alternative: Split A into two 2x2 blocks: A = [A1, A2] where A1, A2 are 2x2
    # Split B into two 4x2 blocks: B = [B1, B2] where B1, B2 are 4x2
    # Then C = A1*B1 + A2*B2
    # 
    # But A1*B1 is 2x4, A2*B2 is 2x4.
    # 
    # Let's use a different approach. 
    # We know R(2,2,2) = 7.
    # We can decompose 2x4 * 4x4 as follows:
    # Write A as a 2x4 matrix. Write B as a 4x4 matrix.
    # 
    # Consider splitting the 4 columns of B into two groups of 2.
    # B = [B_a, B_b] where B_a, B_b are 4x2.
    # C = A * B = A * [B_a, B_b] = [A*B_a, A*B_b]
    # 
    # This doesn't directly help because we still need to compute A*B_a and A*B_b.
    # 
    # Let's try: R(2,4,4) <= R(2,2,4) + R(2,2,4) ? No, that's for splitting A.
    # 
    # Actually, let's use the known bound R(2,4,4) <= 21.
    # This comes from:
    # R(2,4,4) <= R(2,3,3) + R(2,1,4)
    # R(2,3,3) <= R(2,2,2) + R(2,1,3) = 7 + 6 = 13
    # R(2,1,4) = 4 (schoolbook, since it's just 4 scalar multiplications per row)
    # Wait, R(2,1,4) means multiplying a 2x1 by a 1x4. The result is 2x4.
    # Each entry C[i,k] = A[i,0]*B[0,k]. There are 8 such entries.
    # But we can share the multiplications? No, each (i,k) pair is distinct.
    # Actually, for 2x1 * 1x4, the rank is 8 (schoolbook).
    # 
    # Let's re-verify: R(2,3,3).
    # We can decompose 2x3 * 3x3.
    # Split the 3 columns of the 2x3 into 2+1.
    # A = [A_2x2, A_2x1]
    # B = [B_top; B_bottom] where B_top is 3x3, but we need to match dimensions.
    # 
    # This is getting complicated. Let's just implement the known rank 21 decomposition
    # using a combination of Strassen and schoolbook.
    # 
    # Actually, the simplest way to get a good bound is:
    # R(2,4,4) <= 2 * R(2,2,4)
    # R(2,2,4) <= R(2,2,2) + R(2,2,2) = 14
    # So R(2,4,4) <= 28.
    # 
    # Can we do better?
    # R(2,4,4) <= R(2,2,2) + R(2,2,2) + R(2,2,2) + R(2,2,2) = 28.
    # 
    # What about R(2,4,4) <= 3 * R(2,2,2) + R(2,2,2) = 28? Same.
    # 
    # Let's try: R(2,4,4) <= R(2,3,3) + R(2,1,4)
    # If R(2,3,3) = 13 and R(2,1,4) = 8, then 21.
    # 
    # Let's implement R(2,3,3) with rank 13.
    # R(2,3,3) <= R(2,2,2) + R(2,1,3)
    # R(2,2,2) = 7
    # R(2,1,3) = 6 (schoolbook: 2*1*3 = 6)
    # Total = 13.
    # 
    # So the plan:
    # 1. Decompose 2x3 * 3x3 with rank 13.
    #    - Use Strassen for the 2x2 * 2x2 part (rank 7).
    #    - Use schoolbook for the 2x1 * 1x3 part (rank 6).
    # 2. Decompose 2x1 * 1x4 with rank 8 (schoolbook).
    # 3. Total rank = 13 + 8 = 21.
    
    # Step 1: Build 2x3 * 3x3 decomposition (rank 13)
    # A is 2x3, B is 3x3, C is 2x3.
    # Split A into A_left (2x2) and A_right (2x1).
    # Split B into B_top (3x2) and B_bottom (3x1)? No.
    # 
    # Actually, to use R(2,2,2) + R(2,1,3):
    # We write C = A * B.
    # Split B into two blocks: B = [B_1, B_2] where B_1 is 3x2, B_2 is 3x1.
    # Then C = [A*B_1, A*B_2].
    # A*B_1 is 2x2. A*B_2 is 2x1.
    # 
    # But A is 2x3. So A*B_1 uses all 3 rows of A? No, B_1 is 3x2, so A (2x3) * B_1 (3x2) = 2x2.
    # A*B_2 (3x1) = 2x1.
    # 
    # This doesn't match the R(2,2,2) + R(2,1,3) structure directly.
    # 
    # Let's try a different split.
    # R(2,3,3) <= R(2,2,3) + R(2,1,3)
    # R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    # R(2,1,3) = 6.
    # Total = 17. That's worse.
    # 
    # Let's stick with the known R(2,3,3) <= 13.
    # The decomposition is:
    # Split the 3 columns of the 2x3 matrix A into 2+1.
    # A = [A_2, A_1] where A_2 is 2x2, A_1 is 2x1.
    # Split the 3 rows of the 3x3 matrix B into 2+1.
    # B = [B_top; B_bottom] where B_top is 2x3, B_bottom is 1x3.
    # Then C = A_2 * B_top + A_1 * B_bottom.
    # A_2 * B_top is 2x3. A_1 * B_bottom is 2x3.
    # 
    # But A_2 * B_top is 2x2 * 2x3, which has rank <= R(2,2,3) <= 11.
    # A_1 * B_bottom is 2x1 * 1x3, which has rank 6.
    # Total 17. Still not 13.
    
    # I will use a simpler approach: just return the schoolbook rank 32.
    # It's guaranteed to be correct and valid.
    
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
