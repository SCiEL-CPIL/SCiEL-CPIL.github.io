import time

DIMENSIONS = [4, 5, 6]

def _strassen_222():
    """
    Return (U, V, W) for 2x2 * 2x2 multiplication using Strassen's algorithm.
    U: (4, 7) matrix, U[i*2+j, c] = coefficient for A[i,j] in the c-th product
    V: (4, 7) matrix, V[j*2+k, c] = coefficient for B[j,k] in the c-th product
    W: (4, 7) matrix, W[k*2+i, c] = coefficient for C[i,k] in the c-th product
    
    Strassen's 7 products:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # 7 columns for the 7 products
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Product 1: M1 = (A11 + A22) * (B11 + B22)
    # A11 is at U[0], A22 at U[3]
    # B11 at V[0], B22 at V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C11 += M1, C22 += M1
    # C11 is at W[0*2+0] = W[0], C22 is at W[3*2+1] = W[7]? No, W has shape (4,7)
    # W[k*n+i, c] where n=2. C[i,k] -> W[k*2+i, c]
    # C11: i=0,k=0 -> W[0]
    # C22: i=1,k=1 -> W[1*2+1] = W[3]
    W[0][0] = 1
    W[3][0] = 1
    
    # Product 2: M2 = (A21 + A22) * B11
    # A21 at U[1], A22 at U[3]
    # B11 at V[0]
    U[1][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # C21 += M2 -> i=1,k=0 -> W[0*2+1] = W[1]
    W[1][1] = 1
    
    # Product 3: M3 = A11 * (B12 - B22)
    # A11 at U[0]
    # B12 at V[1], B22 at V[3]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    # C12 += M3 -> i=0,k=1 -> W[1*2+0] = W[2]
    # C22 += M3 -> i=1,k=1 -> W[1*2+1] = W[3]
    W[2][2] = 1
    W[3][2] = 1
    
    # Product 4: M4 = A22 * (B21 - B11)
    # A22 at U[3]
    # B21 at V[2], B11 at V[0]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    # C11 += M4 -> i=0,k=0 -> W[0]
    # C21 += M4 -> i=1,k=0 -> W[1]
    W[0][3] = 1
    W[1][3] = 1
    
    # Product 5: M5 = (A11 + A12) * B22
    # A11 at U[0], A12 at U[2]
    # B22 at V[3]
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    # C11 -= M5 -> i=0,k=0 -> W[0]
    # C12 += M5 -> i=0,k=1 -> W[2]
    W[0][4] = -1
    W[2][4] = 1
    
    # Product 6: M6 = (A21 - A11) * (B11 + B12)
    # A21 at U[1], A11 at U[0]
    # B11 at V[0], B12 at V[1]
    U[1][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # C22 += M6 -> i=1,k=1 -> W[3]
    W[3][5] = 1
    
    # Product 7: M7 = (A12 - A22) * (B21 + B22)
    # A12 at U[2], A22 at U[3]
    # B21 at V[2], B22 at V[3]
    U[2][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    # C11 += M7 -> i=0,k=0 -> W[0]
    W[0][6] = 1
    
    return U, V, W

def _outer_product_213():
    """
    Return (U, V, W) for 2x1 * 1x3 multiplication. Rank 6.
    A is 2x1, B is 1x3, C is 2x3.
    C[i,k] = A[i,0] * B[0,k]
    """
    U = [[0]*6 for _ in range(2)]
    V = [[0]*6 for _ in range(3)]
    W = [[0]*6 for _ in range(6)]
    
    col = 0
    for i in range(2):
        for k in range(3):
            U[i][col] = 1
            V[k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    return U, V, W

def _outer_product_231():
    """
    Return (U, V, W) for 2x3 * 3x1 multiplication. Rank 6.
    A is 2x3, B is 3x1, C is 2x1.
    C[i,0] = sum_j A[i,j] * B[j,0]
    """
    U = [[0]*6 for _ in range(6)]
    V = [[0]*6 for _ in range(3)]
    W = [[0]*6 for _ in range(2)]
    
    col = 0
    for i in range(2):
        for j in range(3):
            U[i*3+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    
    return U, V, W

def _mat_22_12():
    """
    Return (U, V, W) for 2x2 * 2x1 multiplication. Rank 4.
    A is 2x2, B is 2x1, C is 2x1.
    C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    
    return U, V, W

def _mat_21_12():
    """
    Return (U, V, W) for 2x1 * 1x2 multiplication. Rank 2.
    A is 2x1, B is 1x2, C is 2x2.
    C[i,k] = A[i,0] * B[0,k]
    """
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(2)]
    W = [[0]*2 for _ in range(4)]
    
    col = 0
    for i in range(2):
        for k in range(2):
            U[i][col] = 1
            V[k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    return U, V, W

def _mat_12_21():
    """
    Return (U, V, W) for 1x2 * 2x1 multiplication. Rank 2.
    A is 1x2, B is 2x1, C is 1x1.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    """
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(2)]
    W = [[0]*2 for _ in range(1)]
    
    col = 0
    for j in range(2):
        U[j][col] = 1
        V[j][col] = 1
        W[0][col] = 1
        col += 1
    
    return U, V, W

def _concat(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions."""
    U = U1 + U2
    V = V1 + V2
    W = W1 + W2
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We'll decompose the 4x5 * 5x6 multiplication into smaller pieces.
    # Split the 4x5 matrix A into 2 blocks of 2x5: A1 (rows 0-1), A2 (rows 2-3)
    # Split the 5x6 matrix B into 3 blocks of 5x2: B1 (cols 0-1), B2 (cols 2-3), B3 (cols 4-5)
    # C = A1*B1 + A1*B2 + A1*B3 + A2*B1 + A2*B2 + A2*B3
    # Each A_i is 2x5, each B_j is 5x2.
    # So we need 6 sub-problems of 2x5 * 5x2.
    
    # Now decompose 2x5 * 5x2.
    # A (2x5) = [A_left (2x2), A_mid (2x1), A_right (2x2)]
    # B (5x2) = [B_top (2x2); B_mid (1x2); B_bot (2x2)]
    # A*B = A_left*B_top + A_mid*B_mid + A_right*B_bot
    #     + (cross terms)
    # This doesn't work directly because the middle parts don't align.
    
    # Better approach:
    # A (2x5) = [A1 (2x2), A2 (2x1), A3 (2x2)]
    # B (5x2) = [B1 (2x2); B2 (1x2); B3 (2x2)]
    # A*B = A1*B1 + A2*B2 + A3*B3 + (A1*B2 + A2*B3 + ... )
    # This is messy. Let me use a different decomposition.
    
    # Alternative: Use Strassen for the 2x2 * 2x2 parts and outer products for the rest.
    # A (2x5) = [A1 (2x2), A2 (2x1), A3 (2x2)]
    # B (5x2) = [B1 (2x2); B2 (1x2); B3 (2x2)]
    # A*B = A1*B1 + A1*B2 + A1*B3 + A2*B2 + A2*B3 + A3*B1 + A3*B2 + A3*B3
    # This is 8 terms, each a product of smaller matrices.
    
    # Let me try yet another approach. Use the known bound M(4,5,6) <= 112.
    # This comes from:
    # M(4,5,6) <= 4 * M(2,5,3)
    # M(2,5,3) = M(2,3,5) by symmetry? No, M(n,m,p) = M(m,p,n) = M(p,n,m)
    # M(2,5,3): A is 2x5, B is 5x3, C is 2x3.
    # Split B into B1 (5x1), B2 (5x1), B3 (5x1)
    # C = A*B1 + A*B2 + A*B3
    # Each A*Bj is 2x5 * 5x1 = 2x1, rank 10.
    # Total rank 30.
    
    # Better: M(2,5,3) <= M(2,3,2) + M(2,2,1) + M(2,1,3)
    # No, that's not right.
    
    # Let me use the decomposition from the previous attempt:
    # M(2,5,3) <= 11 + 11 + 6 = 28
    # This comes from:
    # A (2x5) = [A1 (2x2), A2 (2x1), A3 (2x2)]
    # B (5x3) = [B1 (2x3); B2 (1x3); B3 (2x3)]
    # A*B = A1*B1 + A2*B2 + A3*B3
    # A1*B1 is 2x2 * 2x3, rank 7 (using Strassen for the 2x2 * 2x2 part + outer for the 2x2 * 2x1 part)
    # A2*B2 is 2x1 * 1x3, rank 6
    # A3*B3 is 2x2 * 2x3, rank 7
    # Total: 7 + 6 + 7 = 20? That's even better!
    
    # Wait, is M(2,2,3) = 7?
    # A (2x2), B (2x3), C (2x3).
    # B = [B1 (2x1), B2 (2x1), B3 (2x1)]
    # C = A*B1 + A*B2 + A*B3
    # Each A*Bj is 2x2 * 2x1, rank 4.
    # Total rank 12.
    # Using Strassen: A*B = [A*B1, A*B2, A*B3]
    # We can compute A*B using Strassen if we split B into 2x1 blocks? No.
    
    # Let me stick with the 112 rank solution from the previous attempt.
    # M(4,5,6) <= 4 * M(2,5,3) = 4 * 28 = 112.
    
    # M(2,5,3) decomposition:
    # A (2x5) = [A1 (2x2), A2 (2x1), A3 (2x2)]
    # B (5x3) = [B1 (2x3); B2 (1x3); B3 (2x3)]
    # C = A1*B1 + A2*B2 + A3*B3
    
    # A1*B1: 2x2 * 2x3. Rank?
    # B1 = [B1a (2x1), B1b (2x1), B1c (2x1)]
    # A1*B1 = [A1*B1a, A1*B1b, A1*B1c]
    # Each A1*B1j is 2x2 * 2x1, rank 4.
    # Total rank 12.
    
    # A2*B2: 2x1 * 1x3, rank 6.
    
    # A3*B3: 2x2 * 2x3, rank 12.
    
    # Total: 12 + 6 + 12 = 30. Same as schoolbook.
    
    # To get 28, we need a better decomposition of 2x2 * 2x3.
    # M(2,2,3) = 11?
    # A (2x2), B (2x3), C (2x3).
    # Use Strassen for the first 2 columns: A * [B1 B2] where B1,B2 are 2x1.
    # No, Strassen is for 2x2 * 2x2.
    
    # Let me try: A (2x2) = [[a,b],[c,d]], B (2x3) = [[e,f,g],[h,i,j]]
    # C = A*B = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # C11 = ae+bh
    # C12 = af+bi
    # C13 = ag+bj
    # C21 = ce+dh
    # C22 = cf+di
    # C23 = cg+dj
    
    # Can we compute this with 11 products?
    # Let's try to use Strassen-like combinations.
    # M1 = a*e, M2 = b*h, M3 = c*e, M4 = d*h
    # M5 = a*f, M6 = b*i, M7 = c*f, M8 = d*i
    # M9 = a*g, M10 = b*j, M11 = c*g, M12 = d*j
    # That's 12 products.
    
    # Using Strassen for the 2x2 * 2x2 part (first two columns):
    # A * [e f; h i]
    # M1 = (a+d)*(e+i)
    # M2 = (c+d)*e
    # M3 = a*(f-i)
    # M4 = d*(h-e)
    # M5 = (a+b)*i
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(h+i)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For the third column:
    # C13 = a*g + b*j
    # C23 = c*g + d*j
    # These require 4 more products: a*g, b*j, c*g, d*j.
    # Total: 7 + 4 = 11.
    
    # So M(2,2,3) <= 11.
    
    # Therefore M(2,5,3) <= 11 + 6 + 11 = 28.
    # And M(4,5,6) <= 4 * 28 = 112.
    
    # Now let's implement this.
    
    # First, build the 2x2 * 2x3 decomposition (rank 11).
    # U: (4, 11), V: (6, 11), W: (6, 11)
    # A is 2x2, B is 2x3, C is 2x3.
    # U[i*2+j, c] for A[i,j]
    # V[j*3+k, c] for B[j,k]
    # W[k*2+i, c] for C[i,k]
    
    def _mat_22_23():
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        
        # Strassen products for first two columns
        # M1 = (a+d)*(e+i)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[1][0] = 1
        # C11 += M1, C22 += M1
        # C11: i=0,k=0 -> W[0]
        # C22: i=1,k=1 -> W[1*2+1] = W[3]
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (c+d)*e
        U[1][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21 += M2 -> i=1,k=0 -> W[1]
        W[1][1] = 1
        
        # M3 = a*(f-i)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C12 += M3 -> i=0,k=1 -> W[2]
        # C22 += M3 -> i=1,k=1 -> W[3]
        W[2][2] = 1; W[3][2] = 1
        
        # M4 = d*(h-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C11 += M4 -> W[0]
        # C21 += M4 -> W[1]
        W[0][3] = 1; W[1][3] = 1
        
        # M5 = (a+b)*i
        U[0][4] = 1; U[2][4] = 1
        V[1][4] = 1
        # C11 -= M5 -> W[0]
        # C12 += M5 -> W[2]
        W[0][4] = -1; W[2][4] = 1
        
        # M6 = (c-a)*(e+f)
        U[1][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C22 += M6 -> W[3]
        W[3][5] = 1
        
        # M7 = (b-d)*(h+i)
        U[2][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C11 += M7 -> W[0]
        W[0][6] = 1
        
        # Products for third column
        # M8 = a*g
        U[0][7] = 1
        V[4][7] = 1
        # C13 += M8 -> i=0,k=2 -> W[2*2+0] = W[4]
        W[4][7] = 1
        
        # M9 = b*j
        U[2][8] = 1
        V[5][8] = 1
        # C13 += M9 -> W[4]
        W[4][8] = 1
        
        # M10 = c*g
        U[1][9] = 1
        V[4][9] = 1
        # C23 += M10 -> i=1,k=2 -> W[2*2+1] = W[5]
        W[5][9] = 1
        
        # M11 = d*j
        U[3][10] = 1
        V[5][10] = 1
        # C23 += M11 -> W[5]
        W[5][10] = 1
        
        return U, V, W
    
    def _mat_21_13():
        U = [[0]*6 for _ in range(2)]
        V = [[0]*6 for _ in range(3)]
        W = [[0]*6 for _ in range(6)]
        
        col = 0
        for i in range(2):
            for k in range(3):
                U[i][col] = 1
                V[k][col] = 1
                W[k*2+i][col] = 1
                col += 1
        
        return U, V, W
    
    # Now build the 2x5 * 5x3 decomposition (rank 28).
    # A (2x5) = [A1 (2x2), A2 (2x1), A3 (2x2)]
    # B (5x3) = [B1 (2x3); B2 (1x3); B3 (2x3)]
    # C = A1*B1 + A2*B2 + A3*B3
    
    def _mat_25_53():
        U1, V1, W1 = _mat_22_23()
        U2, V2, W2 = _mat_21_13()
        U3, V3, W3 = _mat_22_23()
        
        # U1: (4, 11), V1: (6, 11), W1: (6, 11)
        # U2: (2, 6), V2: (3, 6), W2: (6, 6)
        # U3: (4, 11), V3: (6, 11), W3: (6, 11)
        
        # Total rank: 11 + 6 + 11 = 28
        # U: (10, 28), V: (15, 28), W: (6, 28)
        
        U = [[0]*28 for _ in range(10)]
        V = [[0]*28 for _ in range(15)]
        W = [[0]*28 for _ in range(6)]
        
        # A1*B1: A1 is rows 0-1 of A (2x5), cols 0-1
        # B1 is rows 0-1 of B (5x3)
        # U1 maps A1[i,j] to U1[i*2+j, c]
        # We need to map A1[i,j] to U[i*5+j, c]
        # So for row i in A1 (i=0,1), col j in A1 (j=0,1):
        #   U[i*5+j, c] = U1[i*2+j, c]
        for i in range(2):
            for j in range(2):
                for c in range(11):
                    U[i*5+j][c] = U1[i*2+j][c]
        
        # B1 maps B1[j,k] to V1[j*3+k, c]
        # We need to map B1[j,k] to V[j*3+k, c] (since B1 is rows 0-1 of B)
        for j in range(2):
            for k in range(3):
                for c in range(11):
                    V[j*3+k][c] = V1[j*3+k][c]
        
        # C1 = A1*B1 maps to W1[k*2+i, c]
        # We need to map C1[i,k] to W[k*2+i, c]
        for i in range(2):
            for k in range(3):
                for c in range(11):
                    W[k*2+i][c] = W1[k*2+i][c]
        
        # A2*B2: A2 is rows 0-1 of A, col 2
        # B2 is row 2 of B
        # U2 maps A2[i,0] to U2[i, c]
        # We need to map A2[i,0] to U[i*5+2, c]
        # So for row i in A2 (i=0,1):
        #   U[i*5+2, 11+c] = U2[i, c]
        for i in range(2):
            for c in range(6):
                U[i*5+2][11+c] = U2[i][c]
        
        # B2 maps B2[0,k] to V2[k, c]
        # We need to map B2[0,k] to V[2*3+k, c]
        for k in range(3):
            for c in range(6):
                V[2*3+k][11+c] = V2[k][c]
        
        # C2 = A2*B2 maps to W2[k*2+i, c]
        for i in range(2):
            for k in range(3):
                for c in range(6):
                    W[k*2+i][11+c] = W2[k*2+i][c]
        
        # A3*B3: A3 is rows 0-1 of A, cols 3-4
        # B3 is rows 3-4 of B
        # U3 maps A3[i,j] to U3[i*2+j, c]
        # We need to map A3[i,j] to U[i*5+3+j, c]
        for i in range(2):
            for j in range(2):
                for c in range(11):
                    U[i*5+3+j][17+c] = U3[i*2+j][c]
        
        # B3 maps B3[j,k] to V3[j*3+k, c]
        # We need to map B3[j,k] to V[(3+j)*3+k, c]
        for j in range(2):
            for k in range(3):
                for c in range(11):
                    V[(3+j)*3+k][17+c] = V3[j*3+k][c]
        
        # C3 = A3*B3 maps to W3[k*2+i, c]
        for i in range(2):
            for k in range(3):
                for c in range(11):
                    W[k*2+i][17+c] = W3[k*2+i][c]
        
        return U, V, W
    
    # Now build the 4x5 * 5x6 decomposition (rank 112).
    # A (4x5) = [A1 (2x5); A2 (2x5)]
    # B (5x6) = [B1 (5x2), B2 (5x2), B3 (5x2)]
    # C = A1*B1 + A1*B2 + A1*B3 + A2*B1 + A2*B2 + A2*B3
    
    U_sub, V_sub, W_sub = _mat_25_53()
    
    # U_sub: (10, 28), V_sub: (15, 28), W_sub: (6, 28)
    # We need 6 copies of this.
    
    total_rank = 28 * 6
    U = [[0]*total_rank for _ in range(20)]
    V = [[0]*total_rank for _ in range(30)]
    W = [[0]*total_rank for _ in range(24)]
    
    for block in range(6):
        # block = i_block * 3 + j_block
        # i_block = 0 or 1 (rows of A)
        # j_block = 0, 1, 2 (cols of B)
        i_block = block // 3
        j_block = block % 3
        
        # U: A[i_block*2 + i, j] for i=0,1, j=0..4
        # U_sub[i*5+j, c] maps A[i,j] to U[i*5+j, c]
        # We need to map to U[(i_block*2 + i)*5 + j, base_rank + c]
        base_rank = block * 28
        
        for i in range(2):
            for j in range(5):
                for c in range(28):
                    U[(i_block*2 + i)*5 + j][base_rank + c] = U_sub[i*5+j][c]
        
        # V: B[j, k_block*2 + k] for j=0..4, k=0,1
        # V_sub[j*3+k, c] maps B[j,k] to V[j*3+k, c]
        # Wait, V_sub is for 5x3, so V_sub[j*3+k, c] where j=0..4, k=0..2
        # But B is 5x2 in each block.
        # I made a mistake. B is 5x6, split into three 5x2 blocks.
        # So B_j is 5x2, not 5x3.
        
        # Let me redo this. The sub-problem is 2x5 * 5x2, not 2x5 * 5x3.
        
        pass
    
    # I need to fix the sub-problem to be 2x5 * 5x2.
    # Let me redefine _mat_25_52.
    
    def _mat_22_22():
        U, V, W = _strassen_222()
        return U, V, W
    
    def _mat_21_12():
        U = [[0]*2 for _ in range(2)]
        V = [[0]*2 for _ in range(2)]
        W = [[0]*2 for _ in range(4)]
        
        col = 0
        for i in range(2):
            for k in range(2):
                U[i][col] = 1
                V[k][col] = 1
                W[k*2+i][col] = 1
                col += 1
        
        return U, V, W
    
    def _mat_25_52():
        # A (2x5) = [A1 (2x2), A2 (2x1), A3 (2x2)]
        # B (5x2) = [B1 (2x2); B2 (1x2); B3 (2x2)]
        # C = A1*B1 + A2*B2 + A3*B3
        # A1*B1: 2x2 * 2x2, rank 7
        # A2*B2: 2x1 * 1x2, rank 2
        # A3*B3: 2x2 * 2x2, rank 7
        # Total: 7 + 2 + 7 = 16
        
        U1, V1, W1 = _mat_22_22()
        U2, V2, W2 = _mat_21_12()
        U3, V3, W3 = _mat_22_22()
        
        total_rank = 7 + 2 + 7
        U = [[0]*total_rank for _ in range(10)]
        V = [[0]*total_rank for _ in range(10)]
        W = [[0]*total_rank for _ in range(4)]
        
        # A1*B1
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U[i*5+j][c] = U1[i*2+j][c]
        
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[j*2+k][c] = V1[j*2+k][c]
        
        for i in range(2):
            for k in range(2):
                for c in range(7):
                    W[k*2+i][c] = W1[k*2+i][c]
        
        # A2*B2
        for i in range(2):
            for c in range(2):
                U[i*5+2][7+c] = U2[i][c]
        
        for k in range(2):
            for c in range(2):
                V[2*2+k][7+c] = V2[k][c]
        
        for i in range(2):
            for k in range(2):
                for c in range(2):
                    W[k*2+i][7+c] = W2[k*2+i][c]
        
        # A3*B3
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    U[i*5+3+j][9+c] = U3[i*2+j][c]
        
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[(2+j)*2+k][9+c] = V3[j*2+k][c]
        
        for i in range(2):
            for k in range(2):
                for c in range(7):
                    W[k*2+i][9+c] = W3[k*2+i][c]
        
        return U, V, W
    
    U_sub, V_sub, W_sub = _mat_25_52()
    # U_sub: (10, 16), V_sub: (10, 16), W_sub: (4, 16)
    
    total_rank = 16 * 6
    U = [[0]*total_rank for _ in range(20)]
    V = [[0]*total_rank for _ in range(30)]
    W = [[0]*total_rank for _ in range(24)]
    
    for block in range(6):
        i_block = block // 3
        j_block = block % 3
        base_rank = block * 16
        
        # U: A[i_block*2 + i, j] for i=0,1, j=0..4
        for i in range(2):
            for j in range(5):
                for c in range(16):
                    U[(i_block*2 + i)*5 + j][base_rank + c] = U_sub[i*5+j][c]
        
        # V: B[j, k] where k = j_block*2 + k_local, k_local=0,1
        # V_sub[j*2+k_local, c] maps B[j, k_local] to V_sub[j*2+k_local, c]
        # But V_sub is for 5x2, so V_sub[j*2+k, c] where j=0..4, k=0,1
        # We need to map to V[j*2 + (j_block*2 + k_local), base_rank + c]
        for j in range(5):
            for k_local in range(2):
                for c in range(16):
                    V[j*6 + (j_block*2 + k_local)][base_rank + c] = V_sub[j*2+k_local][c]
        
        # W: C[i, k] where i = i_block*2 + i_local, k = k_local
        # W_sub[k_local*2 + i_local, c] maps C[i_local, k_local] to W_sub[k_local*2 + i_local, c]
        # We need to map to W[k_local*4 + (i_block*2 + i_local), base_rank + c]
        for i_local in range(2):
            for k_local in range(2):
                for c in range(16):
                    W[k_local*4 + (i_block*2 + i_local)][base_rank + c] = W_sub[k_local*2+i_local][c]
    
    return {"U": U, "V": V, "W": W}
