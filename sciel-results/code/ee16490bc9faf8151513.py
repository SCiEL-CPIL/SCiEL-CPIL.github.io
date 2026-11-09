"""
Tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses recursive Strassen to achieve rank 49 (for 4x4x4) + 16 (for 4x4x1) = 65.
"""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 140

    def verify_decomposition(U, V, W, n, m, p, max_rank=None):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        if max_rank is None:
            max_rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(max_rank):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2_decomposition():
        """
        Strassen decomposition for 2x2*2x2, rank 7.
        U: 4 rows (a11,a12,a21,a22), V: 4 rows (b11,b12,b21,b22), W: 4 rows (c11,c12,c21,c22)
        Each has 7 columns.
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a11+a22)(b11+b22) -> c11 += M1, c12 += M1, c21 -= M1, c22 -= M1
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[1][0] += 1; W[2][0] += -1; W[3][0] += -1
        
        # M2 = (a21+a22)(b12-b11) -> c21 += M2, c22 -= M2
        U[2][1] += 1; U[3][1] += 1
        V[1][1] += 1; V[0][1] += -1
        W[2][1] += 1; W[3][1] += -1
        
        # M3 = a11(b21-b22) -> c12 += M3, c22 -= M3
        U[0][2] += 1
        V[2][2] += 1; V[3][2] += -1
        W[1][2] += 1; W[3][2] += -1
        
        # M4 = a22(b12-b11) -> c11 += M4, c21 -= M4
        U[3][3] += 1
        V[1][3] += 1; V[0][3] += -1
        W[0][3] += 1; W[2][3] += -1
        
        # M5 = (a11+a12)b22 -> c11 += M5, c12 -= M5
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] += 1; W[1][4] += -1
        
        # M6 = (a21-a11)(b11+b12) -> c21 += M6, c22 += M6
        U[2][5] += 1; U[0][5] += -1
        V[0][5] += 1; V[1][5] += 1
        W[2][5] += 1; W[3][5] += 1
        
        # M7 = (a12-a22)(b21+b22) -> c12 += M7, c22 += M7
        U[1][6] += 1; U[3][6] += -1
        V[2][6] += 1; V[3][6] += 1
        W[1][6] += 1; W[3][6] += 1
        
        return U, V, W

    def build_4x4x4_strassen():
        """
        Build 4x4*4x4 decomposition using recursive Strassen. Rank = 49.
        U: 16x49, V: 16x49, W: 16x49
        """
        U2, V2, W2 = strassen_2x2_decomposition()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # Block index maps for 4x4 (row-major):
        # A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
        block_A = {
            '11': [0, 1, 4, 5],
            '12': [2, 3, 6, 7],
            '21': [8, 9, 12, 13],
            '22': [10, 11, 14, 15]
        }
        # B is 4x4: B11: [0,1,4,5], B12: [2,3,6,7], B21: [8,9,12,13], B22: [10,11,14,15]
        block_B = {
            '11': [0, 1, 4, 5],
            '12': [2, 3, 6, 7],
            '21': [8, 9, 12, 13],
            '22': [10, 11, 14, 15]
        }
        # C is 4x4: C11: [0,1,4,5], C12: [2,3,6,7], C21: [8,9,12,13], C22: [10,11,14,15]
        block_C = {
            '11': [0, 1, 4, 5],
            '12': [2, 3, 6, 7],
            '21': [8, 9, 12, 13],
            '22': [10, 11, 14, 15]
        }
        
        # Strassen terms: (A_blocks, B_blocks, C_blocks)
        # Each block entry is {block_name: coefficient}
        strassen_terms = [
            # M1 = (A11+A22)*(B11+B22)
            ({'11': 1, '22': 1}, {'11': 1, '22': 1}, {'11': 1, '12': 1, '21': -1, '22': -1}),
            # M2 = (A21+A22)*(B12-B11)
            ({'21': 1, '22': 1}, {'12': 1, '11': -1}, {'21': 1, '22': -1}),
            # M3 = A11*(B21-B22)
            ({'11': 1}, {'21': 1, '22': -1}, {'12': 1, '22': -1}),
            # M4 = A22*(B12-B11)
            ({'22': 1}, {'12': 1, '11': -1}, {'11': 1, '21': -1}),
            # M5 = (A11+A12)*B22
            ({'11': 1, '12': 1}, {'22': 1}, {'11': 1, '12': -1}),
            # M6 = (A21-A11)*(B11+B12)
            ({'21': 1, '11': -1}, {'11': 1, '12': 1}, {'21': 1, '22': 1}),
            # M7 = (A12-A22)*(B21+B22)
            ({'12': 1, '22': -1}, {'21': 1, '22': 1}, {'12': 1, '22': 1}),
        ]
        
        for s, (A_blocks, B_blocks, C_blocks) in enumerate(strassen_terms):
            for t in range(7):
                col = s * 7 + t
                
                # For U: for each A block with coefficient, add coeff * U2[sub_idx][t]
                for bname, coeff in A_blocks.items():
                    for sub_idx, global_idx in enumerate(block_A[bname]):
                        U[global_idx][col] += coeff * U2[sub_idx][t]
                
                # For V: for each B block with coefficient, add coeff * V2[sub_idx][t]
                for bname, coeff in B_blocks.items():
                    for sub_idx, global_idx in enumerate(block_B[bname]):
                        V[global_idx][col] += coeff * V2[sub_idx][t]
                
                # For W: for each C block with coefficient, add coeff * W2[sub_idx][t]
                for bname, coeff in C_blocks.items():
                    for sub_idx, global_idx in enumerate(block_C[bname]):
                        W[global_idx][col] += coeff * W2[sub_idx][t]
        
        return U, V, W

    def build_4x4x1_schoolbook():
        """
        Build 4x4*4x1 decomposition. Rank = 16.
        U: 16x16, V: 4x16, W: 4x16
        For the k-th column of B (k=0..3) and i-th row of C (i=0..3):
        C[i] = sum_j A[i,j] * B[j]
        """
        rank = 16
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4 + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        
        return U, V, W

    # Build 4x4x4 Strassen (rank 49)
    U44, V44, W44 = build_4x4x4_strassen()
    
    # Verify 4x4x4 part
    if not verify_decomposition(U44, V44, W44, 4, 4, 4, 49):
        # Fallback to schoolbook for 4x4x5
        rank = 80
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(20)]
        W = [[0]*rank for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i*4+j][col] = 1
                    V[j*5+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build 4x4x1 schoolbook for the 5th column (rank 16)
    U41, V41, W41 = build_4x4x1_schoolbook()
    
    # Now combine: total rank = 49 + 16 = 65
    # For 4x4x5:
    # U: 16x65
    # V: 20x65 (4x4 part uses first 4 rows of V44, 5th column uses V41)
    # W: 20x65 (4x4 part uses first 4 rows of W44, 5th column uses W41)
    
    total_rank = 65
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(20)]
    W = [[0]*total_rank for _ in range(20)]
    
    # Copy 4x4x4 part (columns 0-48)
    for i in range(16):
        for c in range(49):
            U[i][c] = U44[i][c]
    for j in range(4):
        for c in range(49):
            V[j][c] = V44[j][c]
    for k in range(4):
        for c in range(49):
            W[k*4 + 0][c] = W44[k*4 + 0][c]  # Wait, W44 is for 4x4 output
    
    # Actually, let me reconsider the indexing.
    # For 4x4x4: U is 16x49, V is 16x49, W is 16x49
    # For 4x4x5: U is 16x65, V is 20x65, W is 20x65
    # The first 49 columns handle the first 4 columns of output (C[:, 0:4])
    # The last 16 columns handle the 5th column of output (C[:, 4])
    
    # For the first 49 columns (4x4x4 part):
    # U[i][c] = U44[i][c] for i in 0..15, c in 0..48
    # V[j][c] = V44[j][c] for j in 0..15, c in 0..48
    #   But V for 4x4x5 has 20 rows (4*5). For the first 4 columns of B, 
    #   V[j*5+k][c] where k=0..3. But V44 has 16 rows for 4x4 B.
    #   The mapping: V44 row j corresponds to B[j//4][j%4] in 4x4.
    #   For 4x4x5, B has 4 rows and 5 cols. The first 4 cols form a 4x4.
    #   So V44[i] (i=0..15) maps to V[i] (i=0..15) in the 4x4x5 V matrix.
    #   V rows 16-19 correspond to B[0][4], B[1][4], B[2][4], B[3][4] (5th column).
    
    # W for 4x4x4: W44 has 16 rows for 4x4 C. 
    # For 4x4x5, W has 20 rows (5*4). W[k*4+i] for k=0..4, i=0..3.
    # W44[i] (i=0..15) maps to W[i] for the first 4 output columns.
    # W rows 16-19 correspond to the 5th output column: W[16+i] for i=0..3.
    
    # Redo:
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(20)]
    W = [[0]*total_rank for _ in range(20)]
    
    # First 49 columns: 4x4x4 Strassen
    for i in range(16):
        for c in range(49):
            U[i][c] = U44[i][c]
    for j in range(16):
        for c in range(49):
            V[j][c] = V44[j][c]
    for k in range(16):
        for c in range(49):
            W[k][c] = W44[k][c]
    
    # Last 16 columns: 4x4x1 schoolbook for 5th column
    for i in range(16):
        for c in range(16):
            U[i][49 + c] = U41[i][c]
    for j in range(4):
        for c in range(16):
            V[16 + j][49 + c] = V41[j][c]  # 5th column of B is rows 16-19 of V
    for k in range(4):
        for c in range(16):
            W[16 + k][49 + c] = W41[k][c]  # 5th column of C is rows 16-19 of W
    
    # Verify
    if verify_decomposition(U, V, W, 4, 4, 5, total_rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = 80
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U[i*4+j][col] = 1
                V[j*5+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
