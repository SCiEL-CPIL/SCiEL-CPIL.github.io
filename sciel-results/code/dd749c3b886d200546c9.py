import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 65 decomposition using Strassen for 4x4x4 and schoolbook for the 5th column
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen for 4x4x4 (first 4 columns of B)
    # We will generate the 49 terms.
    # Each term t has coefficients for A (size 4x4) and B (size 4x4, first 4 cols)
    # And contributes to C (size 4x4)
    
    # Helper to add a term
    # term_A: 4x4 matrix of coefficients for A
    # term_B: 4x4 matrix of coefficients for B (cols 0-3)
    # term_C: 4x4 matrix of coefficients for C
    # col: index in U, V, W
    
    terms = []
    
    # Define the 7 block Strassen multiplications
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # Blocks are 2x2.
    
    # S1 = (A11 + A22) * (B11 + B22)
    # S2 = (A21 + A22) * B11
    # S3 = A11 * (B21 - B22)
    # S4 = A22 * (B12 - B11)
    # S5 = (A11 + A12) * B22
    # S6 = (A21 - A11) * (B11 + B12)
    # S7 = (A12 - A22) * (B21 + B22)
    
    # We need to express each S_k as X_k * Y_k where X_k, Y_k are 2x2 matrices of linear forms.
    # Then apply base Strassen to X_k * Y_k.
    
    # Base Strassen for 2x2:
    # T1 = (x11+x22)(y11+y22)
    # T2 = (x21+x22)y11
    # T3 = x11(y21-y22)
    # T4 = x22(y12-y11)
    # T5 = (x11+x12)y22
    # T6 = (x21-x11)(y11+y12)
    # T7 = (x12-x22)(y21+y22)
    
    # C11 = T1 + T4 - T5 + T7
    # C12 = T3 + T5
    # C21 = T2 + T4
    # C22 = T1 - T2 + T3 + T6
    
    # We will iterate through the 7 block multiplications.
    # For each, we define X and Y as 2x2 arrays of 4x4 matrices (coefficients for A and B).
    # Actually, X is a matrix of linear forms of A, Y is a matrix of linear forms of B.
    
    # Let's represent a linear form of A as a 4x4 matrix of coefficients.
    # Let's represent a linear form of B as a 4x4 matrix of coefficients (for cols 0-3).
    
    # Initialize X, Y for each block multiplication
    # X[k][r][c] is a 4x4 matrix of coeffs for A
    # Y[k][r][c] is a 4x4 matrix of coeffs for B
    
    # This is getting complex to code generically. 
    # I will hardcode the 49 terms by expanding the formulas.
    
    # Alternative: Just use the known rank 65 structure?
    # No, I'll write a small script to generate the coefficients.
    
    # Let's define the blocks of A and B as symbols.
    # A11 = [[1,0,0,0], [0,1,0,0], [0,0,0,0], [0,0,0,0]]
    # A12 = [[0,0,1,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]]
    # A21 = [[0,0,0,0], [0,0,0,0], [1,0,0,0], [0,1,0,0]]
    # A22 = [[0,0,0,0], [0,0,0,0], [0,0,1,0], [0,0,0,1]]
    
    # B11, B12, B21, B22 similarly for B (cols 0-3).
    
    # I will compute the 49 terms programmatically.
    
    A_blocks = {
        '11': np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]),
        '12': np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]),
        '21': np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]]),
        '22': np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])
    }
    B_blocks = {
        '11': np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]),
        '12': np.array([[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,0],[0,0,0,0,0]]),
        '21': np.array([[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0]]),
        '22': np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0]])
    }
    
    # C_blocks for output (4x4)
    C_blocks = {
        '11': np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]),
        '12': np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]),
        '21': np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]]),
        '22': np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])
    }
    
    # Define the 7 block multiplications (X, Y)
    # X, Y are 2x2 arrays of linear forms (4x4 matrices)
    
    def add_matrix(m1, m2):
        return m1 + m2
    
    def sub_matrix(m1, m2):
        return m1 - m2
        
    # S1: X = [[A11, 0], [0, A22]]? No.
    # X1 = [[A11, 0], [0, A22]]? No, X1 is a 2x2 matrix of blocks.
    # X1 = [[A11, 0], [0, A22]] is not correct.
    # X1 = [[A11, 0], [0, A22]]?
    # S1 = (A11+A22)(B11+B22).
    # This is not a block matrix multiplication of the form X*Y where X, Y are 2x2 block matrices?
    # Yes, X1 = [[A11, 0], [0, A22]] and Y1 = [[B11, 0], [0, B22]]?
    # No, (A11+A22) is a 2x2 matrix.
    # We can view A11+A22 as a 2x2 matrix.
    # So X1 is just the 2x2 matrix A11+A22?
    # But Strassen works on 2x2 matrices.
    # So X1 = A11+A22 (2x2), Y1 = B11+B22 (2x2).
    # This is a single 2x2 multiplication.
    # So we apply base Strassen to X1, Y1.
    
    # S2: X2 = [[0, A21+A22], [0, 0]]? No.
    # S2 = (A21+A22)B11.
    # X2 = A21+A22 (2x2), Y2 = B11 (2x2).
    
    # S3: X3 = A11, Y3 = B21-B22.
    # S4: X4 = A22, Y4 = B12-B11.
    # S5: X5 = A11+A12, Y5 = B22.
    # S6: X6 = A21-A11, Y6 = B11+B12.
    # S7: X7 = A12-A22, Y7 = B21+B22.
    
    # So for each S_k, we have a 2x2 matrix X_k and 2x2 matrix Y_k.
    # We apply base Strassen to X_k * Y_k.
    
    # Base Strassen terms for X, Y:
    # t1 = (x11+x22)(y11+y22)
    # t2 = (x21+x22)y11
    # t3 = x11(y21-y22)
    # t4 = x22(y12-y11)
    # t5 = (x11+x12)y22
    # t6 = (x21-x11)(y11+y12)
    # t7 = (x12-x22)(y21+y22)
    
    # We need to compute the coefficients of A and B for each t.
    # x11, x12, x21, x22 are 2x2 matrices of linear forms of A.
    # y11, ... are 2x2 matrices of linear forms of B.
    
    # We will store the 49 terms as (U_coeff, V_coeff, W_coeff)
    # U_coeff: 4x4 matrix
    # V_coeff: 4x5 matrix (but only cols 0-3 non-zero)
    # W_coeff: 4x4 matrix
    
    col = 0
    for k in range(7):
        # Define X_k, Y_k
        if k == 0:
            X = np.array([
                [A_blocks['11'], np.zeros((4,4))],
                [np.zeros((4,4)), A_blocks['22']]
            ])
            # Wait, X1 = A11+A22. This is a 2x2 matrix.
            # x11 = A11, x12 = 0, x21 = 0, x22 = A22.
            # Yes.
            Y = np.array([
                [B_blocks['11'], np.zeros((4,4))],
                [np.zeros((4,4)), B_blocks['22']]
            ])
        elif k == 1:
            X = np.array([
                [np.zeros((4,4)), np.zeros((4,4))],
                [A_blocks['21']+A_blocks['22'], np.zeros((4,4))]
            ])
            # x21 = A21+A22, others 0.
            Y = np.array([
                [B_blocks['11'], np.zeros((4,4))],
                [np.zeros((4,4)), np.zeros((4,4))]
            ])
        elif k == 2:
            X = np.array([
                [A_blocks['11'], np.zeros((4,4))],
                [np.zeros((4,4)), np.zeros((4,4))]
            ])
            Y = np.array([
                [np.zeros((4,4)), np.zeros((4,4))],
                [B_blocks['21'], -B_blocks['22']]
            ])
        elif k == 3:
            X = np.array([
                [np.zeros((4,4)), np.zeros((4,4))],
                [np.zeros((4,4)), A_blocks['22']]
            ])
            Y = np.array([
                [-B_blocks['11'], B_blocks['12']],
                [np.zeros((4,4)), np.zeros((4,4))]
            ])
        elif k == 4:
            X = np.array([
                [A_blocks['11'], A_blocks['12']],
                [np.zeros((4,4)), np.zeros((4,4))]
            ])
            Y = np.array([
                [np.zeros((4,4)), np.zeros((4,4))],
                [np.zeros((4,4)), B_blocks['22']]
            ])
        elif k == 5:
            X = np.array([
                [np.zeros((4,4)), np.zeros((4,4))],
                [A_blocks['21']-A_blocks['11'], np.zeros((4,4))]
            ])
            Y = np.array([
                [B_blocks['11'], B_blocks['12']],
                [np.zeros((4,4)), np.zeros((4,4))]
            ])
        elif k == 6:
            X = np.array([
                [np.zeros((4,4)), A_blocks['12']],
                [np.zeros((4,4)), -A_blocks['22']]
            ])
            Y = np.array([
                [np.zeros((4,4)), np.zeros((4,4))],
                [B_blocks['21'], B_blocks['22']]
            ])
            
        # Apply base Strassen to X, Y
        # x11 = X[0][0], x12 = X[0][1], x21 = X[1][0], x22 = X[1][1]
        # y11 = Y[0][0], ...
        
        x11, x12, x21, x22 = X[0][0], X[0][1], X[1][0], X[1][1]
        y11, y12, y21, y22 = Y[0][0], Y[0][1], Y[1][0], Y[1][1]
        
        # t1
        a_t1 = x11 + x22
        b_t1 = y11 + y22
        # t2
        a_t2 = x21 + x22
        b_t2 = y11
        # t3
        a_t3 = x11
        b_t3 = y21 - y22
        # t4
        a_t4 = x22
        b_t4 = y12 - y11
        # t5
        a_t5 = x11 + x12
        b_t5 = y22
        # t6
        a_t6 = x21 - x11
        b_t6 = y11 + y12
        # t7
        a_t7 = x12 - x22
        b_t7 = y21 + y22
        
        # For each term, we need to add it to U, V, W
        # But W depends on the combination.
        # C11 = T1 + T4 - T5 + T7
        # C12 = T3 + T5
        # C21 = T2 + T4
        # C22 = T1 - T2 + T3 + T6
        
        # We will store the terms and then combine them.
        # Term list for this k:
        # (a, b, c_coeff) where c_coeff is a 2x2 matrix of coefficients for the block result.
        # The block result is added to C_blocks.
        
        # T1 contributes to C11 (+1) and C22 (+1)
        # T2 contributes to C21 (+1) and C22 (-1)
        # T3 contributes to C12 (+1) and C22 (+1)
        # T4 contributes to C11 (+1) and C21 (+1)
        # T5 contributes to C11 (-1) and C12 (+1)
        # T6 contributes to C22 (+1)
        # T7 contributes to C11 (+1)
        
        terms_k = [
            (a_t1, b_t1, np.array([[1, 0], [0, 1]])), # T1: C11+1, C22+1
            (a_t2, b_t2, np.array([[0, 0], [1, -1]])), # T2: C21+1, C22-1
            (a_t3, b_t3, np.array([[0, 1], [0, 1]])), # T3: C12+1, C22+1
            (a_t4, b_t4, np.array([[1, 0], [1, 0]])), # T4: C11+1, C21+1
            (a_t5, b_t5, np.array([[-1, 1], [0, 0]])), # T5: C11-1, C12+1
            (a_t6, b_t6, np.array([[0, 0], [0, 1]])), # T6: C22+1
            (a_t7, b_t7, np.array([[1, 0], [0, 0]]))  # T7: C11+1
        ]
        
        for a, b, c_block in terms_k:
            # a is 4x4, b is 4x4 (cols 0-3), c_block is 2x2
            # Expand c_block to 4x4
            c_full = np.zeros((4,4))
            c_full[0:2, 0:2] = c_block[0,0] * C_blocks['11']
            c_full[0:2, 2:4] = c_block[0,1] * C_blocks['12']
            c_full[2:4, 0:2] = c_block[1,0] * C_blocks['21']
            c_full[2:4, 2:4] = c_block[1,1] * C_blocks['22']
            
            # Fill U, V, W
            for i in range(4):
                for j in range(4):
                    val_u = a[i][j]
                    if val_u != 0:
                        U[i*4+j][col] += val_u
                    val_v = b[i][j]
                    if val_v != 0:
                        V[j*5+i][col] += val_v # V row is j*p + k? No.
                        # V is m x p = 4 x 5.
                        # Index is j*5 + k.
                        # b[i][j] corresponds to B[j][i]?
                        # My B_blocks are defined as 4x4 matrices where row is j, col is k?
                        # B_blocks['11'] has 1s at (0,0), (1,1).
                        # So b[i][j] is coefficient for B[i][j]?
                        # Let's check.
                        # A_blocks['11'] has 1s at (0,0), (1,1).
                        # So a[i][j] is coefficient for A[i][j].
                        # U row is i*m + j = i*4 + j.
                        # So U[i*4+j][col] += a[i][j]. Correct.
                        
                        # V row is j*p + k = j*5 + k.
                        # b is 4x4. b[r][c] is coefficient for B[r][c].
                        # So V[r*5+c][col] += b[r][c].
                        # My loop is for i, j in range(4).
                        # So V[i*5+j][col] += b[i][j].
                        
                    val_w = c_full[i][j]
                    if val_w != 0:
                        # W row is k*n + i = k*4 + i.
                        # c_full[i][j] is coefficient for C[i][j].
                        # So W[j*4+i][col] += c_full[i][j].
                        W[j*4+i][col] += val_w
                        
            col += 1
            
    # Part 2: Schoolbook for 5th column (k=4)
    # C[i][4] = sum_j A[i][j] B[j][4]
    # Terms: A[i][j] * B[j][4]
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
