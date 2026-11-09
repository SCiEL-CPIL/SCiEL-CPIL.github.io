"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build rank-7 Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    U_2x2x2 = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 1, 1]
    ]
    V_2x2x2 = [
        [1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1]
    ]
    W_2x2x2 = [
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1]
    ]
    
    # Build 4x4x4 decomposition using Strassen (rank 49)
    # Split 4x4 into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B = [[B11, B12], [B21, B22]] each 2x2
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
    
    # For 4x4x4: rank = 7 * 7 = 49
    # We need to build U, V, W matrices of size 16x49
    
    # Each 2x2x2 multiplication uses 7 scalar multiplications
    # For 4x4x4, we have 7 such block multiplications, each of size 2x2x2
    
    # Let's build the 4x4x4 decomposition explicitly
    U_4x4x4 = [[0]*49 for _ in range(16)]
    V_4x4x4 = [[0]*49 for _ in range(16)]
    W_4x4x4 = [[0]*49 for _ in range(16)]
    
    # Block indices: row_block (0,1), col_block (0,1), inner (0,1)
    # For a 2x2 block at position (rb, cb), the 4 elements are at:
    # (rb*2+0, cb*2+0), (rb*2+0, cb*2+1), (rb*2+1, cb*2+0), (rb*2+1, cb*2+1)
    # In flattened U index: row*4 + col -> (rb*2+i)*4 + (cb*2+j) = rb*8 + i*4 + cb*2 + j
    
    # Strassen block multipliers for 4x4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each multiplier Mk, we have a 2x2x2 multiplication
    # The 7 scalar multiplications within Mk are indexed 0..6
    
    # M1: (A11+A22)(B11+B22)
    # A part: A11 and A22 with coefficients
    # A11: (0,0) block -> indices in U: 0,1,4,5 (for 2x2)
    # A22: (1,1) block -> indices in U: 8,9,12,13
    # B part: B11 and B12 -> indices in V: 0,1,4,5 (B11) and 2,3,6,7 (B12)
    # Wait, let me be more careful.
    
    # For matrix A (4x4), flattened as row-major: A[r*4+c]
    # Block A11: rows 0-1, cols 0-1 -> A[0],A[1],A[4],A[5]
    # Block A12: rows 0-1, cols 2-3 -> A[2],A[3],A[6],A[7]
    # Block A21: rows 2-3, cols 0-1 -> A[8],A[9],A[12],A[13]
    # Block A22: rows 2-3, cols 2-3 -> A[10],A[11],A[14],A[15]
    
    # For matrix B (4x4), flattened as row-major: B[r*4+c]
    # Block B11: rows 0-1, cols 0-1 -> B[0],B[1],B[4],B[5]
    # Block B12: rows 0-1, cols 2-3 -> B[2],B[3],B[6],B[7]
    # Block B21: rows 2-3, cols 0-1 -> B[8],B[9],B[12],B[13]
    # Block B22: rows 2-3, cols 2-3 -> B[10],B[11],B[14],B[15]
    
    # For each Strassen multiplier Mk, we perform a 2x2x2 multiplication
    # The 2x2x2 rank-7 decomposition has 7 scalar multiplications
    # For the k-th multiplier (k=0..6 for M1..M7), the scalar mult index is 7*k + s (s=0..6)
    
    # Let me define the Strassen multipliers more carefully:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For M1: A_factor = A11 + A22, B_factor = B11 + B22
    # A11 indices: [0,1,4,5], A22 indices: [10,11,14,15]
    # B11 indices: [0,1,4,5], B22 indices: [10,11,14,15]
    
    # For each of the 7 scalar multiplications in the 2x2x2 rank-7 decomp:
    # U_2x2x2[row][s] * V_2x2x2[col][s] contributes to W_2x2x2[out][s]
    
    # The 2x2x2 decomp: for input 2x2 matrices X and Y, output Z = XY
    # X flattened: [x00, x01, x10, x11]
    # Y flattened: [y00, y01, y10, y11]
    # Z flattened: [z00, z01, z10, z11]
    
    # U_2x2x2 maps X to 7 scalar values
    # V_2x2x2 maps Y to 7 scalar values
    # W_2x2x2 maps the 7 products to Z
    
    # Let's use a standard rank-7 Strassen decomposition:
    # S1 = x00, t1 = y00
    # S2 = x11, t2 = y11
    # S3 = x01, t3 = y10
    # S4 = x10, t4 = y01
    # S5 = x00+x11, t5 = y00+y11
    # S6 = x10+x11, t6 = y01
    # S7 = x00-x11, t7 = y10+y11
    
    # z00 = S1*t1 + S5*t5 - S6*t6 + S7*t7
    # z01 = S3*t3 + S5*t5
    # z10 = S4*t4 + S6*t6
    # z11 = S1*t1 - S2*t2 + S3*t3 + S7*t7
    
    # Wait, let me use a cleaner formulation. Standard Strassen:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # For 2x2x2, the rank-7 decomposition uses these 7 multipliers directly
    # So U_2x2x2, V_2x2x2, W_2x2x2 are:
    # U: maps [a11,a12,a21,a22] to the 7 scalar factors on the A side
    # V: maps [b11,b12,b21,b22] to the 7 scalar factors on the B side
    # W: maps the 7 products to [c11,c12,c21,c22]
    
    # A side factors:
    # m1: a11 + a22 -> [1,0,0,1]
    # m2: a21 + a22 -> [0,0,1,1]
    # m3: a11 -> [1,0,0,0]
    # m4: a22 -> [0,0,0,1]
    # m5: a11 + a12 -> [1,1,0,0]
    # m6: a21 - a11 -> [-1,0,1,0]
    # m7: a12 - a22 -> [0,1,0,-1]
    
    # B side factors:
    # m1: b11 + b22 -> [1,0,0,1]
    # m2: b11 -> [1,0,0,0]
    # m3: b12 - b22 -> [0,1,0,-1]
    # m4: b21 - b11 -> [-1,0,1,0]
    # m5: b22 -> [0,0,0,1]
    # m6: b11 + b12 -> [1,1,0,0]
    # m7: b21 + b22 -> [0,0,1,1]
    
    # C side (W):
    # c11 = m1 + m4 - m5 + m7 -> [1,0,0,1,0,0,1]
    # c12 = m3 + m5 -> [0,0,1,0,1,0,0]
    # c21 = m2 + m4 -> [0,1,0,1,0,0,0]
    # c22 = m1 - m2 + m3 + m6 -> [1,-1,1,0,0,1,0]
    
    U_2x2x2 = [
        [1, 0, 1, 0, 1, -1, 0],   # a11
        [0, 0, 0, 0, 1, 0, 1],    # a12
        [0, 1, 0, 0, 0, 1, 0],    # a21
        [1, 1, 0, 1, 0, 0, -1]    # a22
    ]
    V_2x2x2 = [
        [1, 1, 0, -1, 0, 1, 0],   # b11
        [0, 0, 1, 0, 0, 1, 0],    # b12
        [0, 0, 0, 1, 0, 0, 1],    # b21
        [1, 0, -1, 0, 1, 0, 1]    # b22
    ]
    W_2x2x2 = [
        [1, 0, 0, 1, 0, 0, 1],    # c11
        [0, 0, 1, 0, 1, 0, 0],    # c12
        [0, 1, 0, 1, 0, 0, 0],    # c21
        [1, -1, 1, 0, 0, 1, 0]    # c22
    ]
    
    # Now build 4x4x4 decomposition
    # For each Strassen multiplier Mk (k=0..6), we have a 2x2x2 multiplication
    # The scalar mult index for Mk, s-th component is 7*k + s
    
    # Define the A-side block contributions for each Mk:
    # M1: A11 + A22
    # M2: A21 + A22
    # M3: A11
    # M4: A22
    # M5: A11 + A12
    # M6: A21 - A11
    # M7: A12 - A22
    
    # A11: indices [0,1,4,5] (a11,a12,a21,a22 of block)
    # A12: indices [2,3,6,7]
    # A21: indices [8,9,12,13]
    # A22: indices [10,11,14,15]
    
    # For each block position, the 2x2 elements map to U_2x2x2 rows 0,1,2,3
    # Block A11: element (i,j) in block -> global index (0*2+i)*4 + (0*2+j) = i*4+j
    #   a11->0, a12->1, a21->4, a22->5
    # Block A12: element (i,j) in block -> global index (0*2+i)*4 + (1*2+j) = i*4+2+j
    #   a11->2, a12->3, a21->6, a22->7
    # Block A21: element (i,j) in block -> global index (1*2+i)*4 + (0*2+j) = 8+i*4+j
    #   a11->8, a12->9, a21->12, a22->13
    # Block A22: element (i,j) in block -> global index (1*2+i)*4 + (1*2+j) = 8+i*4+2+j
    #   a11->10, a12->11, a21->14, a22->15
    
    # A-side block coefficient patterns for each Mk:
    # M1: A11 + A22 -> blocks [(A11, +1), (A22, +1)]
    # M2: A21 + A22 -> blocks [(A21, +1), (A22, +1)]
    # M3: A11 -> blocks [(A11, +1)]
    # M4: A22 -> blocks [(A22, +1)]
    # M5: A11 + A12 -> blocks [(A11, +1), (A12, +1)]
    # M6: A21 - A11 -> blocks [(A21, +1), (A11, -1)]
    # M7: A12 - A22 -> blocks [(A12, +1), (A22, -1)]
    
    # B-side block coefficient patterns for each Mk:
    # M1: B11 + B22 -> blocks [(B11, +1), (B22, +1)]
    # M2: B11 -> blocks [(B11, +1)]
    # M3: B12 - B22 -> blocks [(B12, +1), (B22, -1)]
    # M4: B21 - B11 -> blocks [(B21, +1), (B11, -1)]
    # M5: B22 -> blocks [(B22, +1)]
    # M6: B11 + B12 -> blocks [(B11, +1), (B12, +1)]
    # M7: B21 + B22 -> blocks [(B21, +1), (B22, +1)]
    
    # C-side: c11, c12, c21, c22 each get contributions from M1..M7
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # For each output block (c11, c12, c21, c22), the 2x2 elements map to W indices
    # C11: indices [0,1,4,5]
    # C12: indices [2,3,6,7]
    # C21: indices [8,9,12,13]
    # C22: indices [10,11,14,15]
    
    # Build U_4x4x4, V_4x4x4, W_4x4x4
    U_4x4x4 = [[0]*49 for _ in range(16)]
    V_4x4x4 = [[0]*49 for _ in range(16)]
    W_4x4x4 = [[0]*49 for _ in range(16)]
    
    # A-side block patterns
    A_blocks = {
        'A11': ([0,1,4,5], [1,1,1,1]),
        'A12': ([2,3,6,7], [1,1,1,1]),
        'A21': ([8,9,12,13], [1,1,1,1]),
        'A22': ([10,11,14,15], [1,1,1,1])
    }
    # For each block, the 4 elements correspond to a11,a12,a21,a22 of the 2x2 block
    # U_2x2x2[row] gives coefficients for that element
    
    A_side_patterns = [
        [('A11', 1), ('A22', 1)],    # M1
        [('A21', 1), ('A22', 1)],    # M2
        [('A11', 1)],                # M3
        [('A22', 1)],                # M4
        [('A11', 1), ('A12', 1)],    # M5
        [('A21', 1), ('A11', -1)],   # M6
        [('A12', 1), ('A22', -1)]    # M7
    ]
    
    B_side_patterns = [
        [('B11', 1), ('B22', 1)],    # M1
        [('B11', 1)],                # M2
        [('B12', 1), ('B22', -1)],   # M3
        [('B21', 1), ('B11', -1)],   # M4
        [('B22', 1)],                # M5
        [('B11', 1), ('B12', 1)],    # M6
        [('B21', 1), ('B22', 1)]     # M7
    ]
    
    # B block indices
    B_blocks = {
        'B11': ([0,1,4,5], [1,1,1,1]),
        'B12': ([2,3,6,7], [1,1,1,1]),
        'B21': ([8,9,12,13], [1,1,1,1]),
        'B22': ([10,11,14,15], [1,1,1,1])
    }
    
    # Fill U and V for 4x4x4
    for k in range(7):  # M1..M7
        # A side
        for block_name, coeff in A_side_patterns[k]:
            indices, ones = A_blocks[block_name]
            for elem_idx, global_idx in enumerate(indices):
                # elem_idx 0,1,2,3 corresponds to a11,a12,a21,a22
                # U_2x2x2[elem_idx][s] is the coefficient
                for s in range(7):
                    U_4x4x4[global_idx][7*k + s] += coeff * U_2x2x2[elem_idx][s]
        
        # B side
        for block_name, coeff in B_side_patterns[k]:
            indices, ones = B_blocks[block_name]
            for elem_idx, global_idx in enumerate(indices):
                for s in range(7):
                    V_4x4x4[global_idx][7*k + s] += coeff * V_2x2x2[elem_idx][s]
    
    # Fill W for 4x4x4
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    C_side_patterns = [
        # (c_block, list of (M_index, coeff))
        ('C11', [(0, 1), (3, 1), (4, -1), (6, 1)]),   # M1 + M4 - M5 + M7
        ('C12', [(2, 1), (4, 1)]),                      # M3 + M5
        ('C21', [(1, 1), (3, 1)]),                      # M2 + M4
        ('C22', [(0, 1), (1, -1), (2, 1), (5, 1)])      # M1 - M2 + M3 + M6
    ]
    
    C_blocks = {
        'C11': [0,1,4,5],
        'C12': [2,3,6,7],
        'C21': [8,9,12,13],
        'C22': [10,11,14,15]
    }
    
    for c_block_name, m_contribs in C_side_patterns:
        c_indices = C_blocks[c_block_name]
        for elem_idx, global_idx in enumerate(c_indices):
            # W_2x2x2[elem_idx][s] is the coefficient for the s-th product in a 2x2x2
            for m_idx, coeff in m_contribs:
                for s in range(7):
                    W_4x4x4[global_idx][7*m_idx + s] += coeff * W_2x2x2[elem_idx][s]
    
    # Now build 4x4x8 by combining two 4x4x4 decompositions
    # A is 4x4, B is 4x8
    # Split B into B_left (4x4) and B_right (4x4)
    # C_left = A * B_left, C_right = A * B_right
    # C = [C_left, C_right]
    
    # For 4x4x8:
    # U: 4*4 = 16 rows, rank = 98
    # V: 4*8 = 32 rows, rank = 98
    # W: 8*4 = 32 rows, rank = 98
    
    # U is the same for both halves (since A is the same)
    # V_left uses V_4x4x4 for B_left (cols 0-3 of B)
    # V_right uses V_4x4x4 for B_right (cols 4-7 of B)
    # W_left uses W_4x4x4 for C_left (cols 0-3 of C)
    # W_right uses W_4x4x4 for C_right (cols 4-7 of C)
    
    rank = 98
    U_4x4x8 = [[0]*rank for _ in range(16)]
    V_4x4x8 = [[0]*rank for _ in range(32)]
    W_4x4x8 = [[0]*rank for _ in range(32)]
    
    # U is the same as U_4x4x4
    for i in range(16):
        for c in range(49):
            U_4x4x8[i][c] = U_4x4x4[i][c]
            U_4x4x8[i][c + 49] = U_4x4x4[i][c]
    
    # V: first 16 rows use V_4x4x4 for B_left, next 16 rows use V_4x4x4 for B_right
    for i in range(16):
        for c in range(49):
            V_4x4x8[i][c] = V_4x4x4[i][c]           # B_left
            V_4x4x8[i + 16][c + 49] = V_4x4x4[i][c]  # B_right
    
    # W: first 16 rows use W_4x4x4 for C_left, next 16 rows use W_4x4x4 for C_right
    # C is 4x8, W has 8*4=32 rows
    # W[k*n+i, c] where k is col of C, i is row of C
    # C_left: k=0..3, i=0..3 -> W[0..15]
    # C_right: k=4..7, i=0..3 -> W[16..31]
    for i in range(16):
        for c in range(49):
            W_4x4x8[i][c] = W_4x4x4[i][c]           # C_left
            W_4x4x8[i + 16][c + 49] = W_4x4x4[i][c]  # C_right
    
    return {"U": U_4x4x8, "V": V_4x4x8, "W": W_4x4x8}
# EVOLVE-BLOCK-END

