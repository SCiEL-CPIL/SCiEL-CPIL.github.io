import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # We split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # AB = A B1 + A B2 + A B3
    # Rank(A B1) = 49 (Strassen)
    # Rank(A B2) = 49 (Strassen)
    # Rank(A B3) = 16 (Schoolbook)
    # Total Rank = 114
    
    U1, V1, W1 = strassen_4x4x4()
    U2, V2, W2 = strassen_4x4x4()
    U3, V3, W3 = schoolbook_4x4x1()
    
    # Concatenate U, V, W
    # U is (16, rank)
    # V is (4*9, rank) ? No, V is (m*p, rank).
    # Wait, the interface is:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # For A B1:
    # A is 4x4. B1 is 4x4.
    # U1 is (16, 49).
    # V1 is (16, 49). But V should be (m*p, rank) = (4*9, rank).
    # B1 is the first 4 columns of B.
    # So V1 should be placed in the first 4 columns of B, i.e., rows 0..3 of V?
    # No, V is (m*p, rank). m=4, p=9.
    # B is 4x9.
    # B1 is 4x4. It corresponds to B[:, 0:4].
    # In the flattened V matrix (size 36x rank), the rows corresponding to B1 are 0..3?
    # No, V is (m*p, rank). The rows of V correspond to the entries of B.
    # B is 4x9. Flattened size 36.
    # B1 is 4x4. It uses the first 4 columns of B.
    # So the rows of V corresponding to B1 are 0..15?
    # Let's check the flattening order.
    # Usually row-major. B[i, j] is at index i*9 + j.
    # B1[i, j] = B[i, j] for j in 0..3.
    # So indices are i*9 + j for i in 0..3, j in 0..3.
    # This is not a contiguous block in row-major order.
    # Indices: 0,1,2,3, 9,10,11,12, 18,19,20,21, 27,28,29,30.
    
    # This makes concatenation tricky.
    # We need to place the V1 entries into the correct rows of the final V matrix.
    
    # Let's create the full U, V, W matrices.
    rank_total = 49 + 49 + 16
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Add Strassen 1 (for B1)
    for c in range(49):
        for r in range(16):
            U[r][c] = U1[r][c]
        # V1 is 16x49.
        # We need to map V1 rows to V rows.
        # V1 row index k corresponds to B1 entry.
        # B1 is 4x4. Flattened index k = i*4 + j.
        # B1[i, j] = B[i, j].
        # V row index = i*9 + j.
        for r in range(16):
            i = r // 4
            j = r % 4
            v_row = i * 9 + j
            V[v_row][c] = V1[r][c]
        # W1 is 16x49.
        # W1 row index k corresponds to C1 entry.
        # C1 is 4x4. Flattened index k = i*4 + j.
        # C1[i, j] = C[i, j].
        # W row index = j*4 + i? No, W is (p*n, rank).
        # W[k*n+i, c].
        # C is 4x9. C1 is 4x4.
        # C1[i, j] = C[i, j].
        # W index = j*4 + i.
        for r in range(16):
            i = r // 4
            j = r % 4
            w_row = j * 4 + i
            W[w_row][c] = W1[r][c]
            
    # Add Strassen 2 (for B2)
    offset = 49
    for c in range(49):
        for r in range(16):
            U[r][c + offset] = U2[r][c]
        # B2 is 4x4. It corresponds to B[:, 4:8].
        # B2[i, j] = B[i, j+4].
        # V row index = i*9 + (j+4).
        for r in range(16):
            i = r // 4
            j = r % 4
            v_row = i * 9 + (j + 4)
            V[v_row][c + offset] = V2[r][c]
        # C2 is 4x4. C2[i, j] = C[i, j+4].
        # W index = (j+4)*4 + i.
        for r in range(16):
            i = r // 4
            j = r % 4
            w_row = (j + 4) * 4 + i
            W[w_row][c + offset] = W2[r][c]
            
    # Add Schoolbook (for B3)
    offset = 98
    for c in range(16):
        for r in range(16):
            U[r][c + offset] = U3[r][c]
        # B3 is 4x1. It corresponds to B[:, 8].
        # B3[i, 0] = B[i, 8].
        # V row index = i*9 + 8.
        for r in range(4): # V3 is 4x16
            i = r
            v_row = i * 9 + 8
            V[v_row][c + offset] = V3[r][c]
        # C3 is 4x1. C3[i, 0] = C[i, 8].
        # W index = 8*4 + i.
        for r in range(4): # W3 is 4x16
            i = r
            w_row = 8 * 4 + i
            W[w_row][c + offset] = W3[r][c]
            
    return {"U": U, "V": V, "W": W}

def strassen_4x4x4():
    # Returns U, V, W for 4x4x4 multiplication using Strassen
    # U: 16x49, V: 16x49, W: 16x49
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # 7 sub-products, each 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # We need the 2x2x2 Strassen U, V, W
    U2, V2, W2 = strassen_2x2x2()
    
    # Helper to add a sub-product
    # a_blocks: list of (block_index, sign) for A
    # b_blocks: list of (block_index, sign) for B
    # c_blocks: list of (block_index, sign) for C
    # offset: column offset in U, V, W
    def add_sub_product(a_blocks, b_blocks, c_blocks, offset):
        for k in range(7):
            col = offset + k
            for (ai, sign_a) in a_blocks:
                for r in range(4):
                    # ai is 0..3 for A blocks
                    # r is 0..3 for 2x2 block
                    a_row = ai * 4 + r
                    U[a_row][col] += sign_a * U2[r][k]
            for (bi, sign_b) in b_blocks:
                for r in range(4):
                    b_row = bi * 4 + r
                    V[b_row][col] += sign_b * V2[r][k]
            for (ci, sign_c) in c_blocks:
                for r in range(4):
                    c_row = ci * 4 + r
                    W[c_row][col] += sign_c * W2[r][k]
                    
    # M1
    add_sub_product([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)], 0)
    # M2
    add_sub_product([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)], 7)
    # M3
    add_sub_product([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)], 14)
    # M4
    add_sub_product([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)], 21)
    # M5
    add_sub_product([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)], 28)
    # M6
    add_sub_product([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 35)
    # M7
    add_sub_product([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)], 42)
    
    return U, V, W

def strassen_2x2x2():
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def schoolbook_4x4x1():
    U = [[0] * 16 for _ in range(16)]
    V = [[0] * 16 for _ in range(4)]
    W = [[0] * 16 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            U[i * 4 + j][c] = 1
            V[j][c] = 1
            W[i][c] = 1
            
    return U, V, W
