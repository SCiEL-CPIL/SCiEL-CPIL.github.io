import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Strategy: Treat 4x4*4x8 as two independent 4x4*4x4 multiplications (left half and right half of B).
    Use the known optimal rank-49 decomposition for 4x4x4 matrix multiplication.
    Total rank = 49 + 49 = 98.
    
    The rank-49 decomposition for 4x4x4 is constructed using a hierarchical Strassen approach:
    - Split 4x4 matrices into four 2x2 blocks
    - Apply Strassen's algorithm (rank 7) at the block level
    - This gives 7*7 = 49 scalar multiplications
    
    For the 4x4x8 case:
    - U has shape (16, 98): A factors for both halves
    - V has shape (32, 98): B factors (16 for left half, 16 for right half)
    - W has shape (32, 98): C factors (16 for left half, 16 for right half)
    """
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Strassen's algorithm for 2x2x2 matrix multiplication (rank 7)
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2x2, the factors are:
    # U_222: 4x7 (A factors)
    # V_222: 4x7 (B factors)  
    # W_222: 4x7 (C factors)
    
    # Indexing for 2x2: element (i,j) -> index i*2+j
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    U_222 = [[0]*7 for _ in range(4)]
    V_222 = [[0]*7 for _ in range(4)]
    W_222 = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11=index 0, A22=index 3, B11=index 0, B22=index 3
    U_222[0][0] = 1  # A11
    U_222[3][0] = 1  # A22
    V_222[0][0] = 1  # B11
    V_222[3][0] = 1  # B22
    # C11 += M1, C22 += M1
    W_222[0][0] = 1  # C11
    W_222[3][0] = 1  # C22
    
    # M2 = (A21 + A22)B11
    # A21=index 2, A22=index 3, B11=index 0
    U_222[2][1] = 1  # A21
    U_222[3][1] = 1  # A22
    V_222[0][1] = 1  # B11
    # C21 += M2, C22 -= M2
    W_222[2][1] = 1  # C21
    W_222[3][1] = -1 # C22
    
    # M3 = A11(B12 - B22)
    # A11=index 0, B12=index 1, B22=index 3
    U_222[0][2] = 1  # A11
    V_222[1][2] = 1  # B12
    V_222[3][2] = -1 # B22
    # C12 += M3, C22 += M3
    W_222[1][2] = 1  # C12
    W_222[3][2] = 1  # C22
    
    # M4 = A22(B21 - B11)
    # A22=index 3, B21=index 2, B11=index 0
    U_222[3][3] = 1  # A22
    V_222[2][3] = 1  # B21
    V_222[0][3] = -1 # B11
    # C11 += M4, C21 += M4
    W_222[0][3] = 1  # C11
    W_222[2][3] = 1  # C21
    
    # M5 = (A11 + A12)B22
    # A11=index 0, A12=index 1, B22=index 3
    U_222[0][4] = 1  # A11
    U_222[1][4] = 1  # A12
    V_222[3][4] = 1  # B22
    # C11 -= M5, C12 += M5
    W_222[0][4] = -1 # C11
    W_222[1][4] = 1  # C12
    
    # M6 = (A21 - A11)(B11 + B12)
    # A21=index 2, A11=index 0, B11=index 0, B12=index 1
    U_222[2][5] = 1  # A21
    U_222[0][5] = -1 # A11
    V_222[0][5] = 1  # B11
    V_222[1][5] = 1  # B12
    # C22 += M6
    W_222[3][5] = 1  # C22
    
    # M7 = (A12 - A22)(B21 + B22)
    # A12=index 1, A22=index 3, B21=index 2, B22=index 3
    U_222[1][6] = 1  # A12
    U_222[3][6] = -1 # A22
    V_222[2][6] = 1  # B21
    V_222[3][6] = 1  # B22
    # C11 += M7
    W_222[0][6] = 1  # C11
    
    # Now build the 4x4x4 decomposition using hierarchical Strassen
    # Split 4x4 into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # Block indices: A00=elements 0-3, A01=elements 4-7, A10=elements 8-11, A11=elements 12-15
    # Similarly for B and C
    
    # Apply Strassen at the block level:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix product, which uses 7 scalar multiplications
    # So total scalar multiplications = 7 * 7 = 49
    
    # Build U_444, V_444, W_444 with shape (16, 49)
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    # For each block-level Strassen product Mi (i=0..6), we have 7 scalar products
    # Scalar product index s (s=0..6) corresponds to column i*7 + s
    
    # Block A00: elements 0-3, A01: elements 4-7, A10: elements 8-11, A11: elements 12-15
    # Block B00: elements 0-3, B01: elements 4-7, B10: elements 8-11, B11: elements 12-15
    # Block C00: elements 0-3, C01: elements 4-7, C10: elements 8-11, C11: elements 12-15
    
    # For each block-level product Mi, determine which A and B blocks are used
    # and with what coefficients
    
    # M1 = (A00 + A11)(B00 + B11)
    # A blocks: A00 (coeff 1), A11 (coeff 1)
    # B blocks: B00 (coeff 1), B11 (coeff 1)
    # C contribution: C00 (coeff 1), C11 (coeff 1)
    
    # M2 = (A10 + A11)B00
    # A blocks: A10 (coeff 1), A11 (coeff 1)
    # B blocks: B00 (coeff 1)
    # C contribution: C10 (coeff 1), C11 (coeff -1)
    
    # M3 = A00(B01 - B11)
    # A blocks: A00 (coeff 1)
    # B blocks: B01 (coeff 1), B11 (coeff -1)
    # C contribution: C01 (coeff 1), C11 (coeff 1)
    
    # M4 = A11(B10 - B00)
    # A blocks: A11 (coeff 1)
    # B blocks: B10 (coeff 1), B00 (coeff -1)
    # C contribution: C00 (coeff 1), C10 (coeff 1)
    
    # M5 = (A00 + A01)B11
    # A blocks: A00 (coeff 1), A01 (coeff 1)
    # B blocks: B11 (coeff 1)
    # C contribution: C00 (coeff -1), C01 (coeff 1)
    
    # M6 = (A10 - A00)(B00 + B01)
    # A blocks: A10 (coeff 1), A00 (coeff -1)
    # B blocks: B00 (coeff 1), B01 (coeff 1)
    # C contribution: C11 (coeff 1)
    
    # M7 = (A01 - A11)(B10 + B11)
    # A blocks: A01 (coeff 1), A11 (coeff -1)
    # B blocks: B10 (coeff 1), B11 (coeff 1)
    # C contribution: C00 (coeff 1)
    
    # Define the block-level Strassen structure
    # For each Mi (i=0..6):
    #   a_blocks: list of (block_idx, coeff) for A
    #   b_blocks: list of (block_idx, coeff) for B
    #   c_blocks: list of (block_idx, coeff) for C
    
    block_structure = [
        # M1
        {'a': [(0, 1), (3, 1)], 'b': [(0, 1), (3, 1)], 'c': [(0, 1), (3, 1)]},
        # M2
        {'a': [(2, 1), (3, 1)], 'b': [(0, 1)], 'c': [(2, 1), (3, -1)]},
        # M3
        {'a': [(0, 1)], 'b': [(1, 1), (3, -1)], 'c': [(1, 1), (3, 1)]},
        # M4
        {'a': [(3, 1)], 'b': [(2, 1), (0, -1)], 'c': [(0, 1), (2, 1)]},
        # M5
        {'a': [(0, 1), (1, 1)], 'b': [(3, 1)], 'c': [(0, -1), (1, 1)]},
        # M6
        {'a': [(2, 1), (0, -1)], 'b': [(0, 1), (1, 1)], 'c': [(3, 1)]},
        # M7
        {'a': [(1, 1), (3, -1)], 'b': [(2, 1), (3, 1)], 'c': [(0, 1)]},
    ]
    
    # Block element ranges:
    # Block 0: elements 0-3
    # Block 1: elements 4-7
    # Block 2: elements 8-11
    # Block 3: elements 12-15
    
    def block_elements(block_idx):
        return list(range(block_idx * 4, (block_idx + 1) * 4))
    
    # Build the 4x4x4 factors
    for mi, structure in enumerate(block_structure):
        for s in range(7):  # 7 scalar products per block-level product
            col = mi * 7 + s
            
            # For A factors
            for block_idx, coeff in structure['a']:
                elems = block_elements(block_idx)
                for local_idx, elem in enumerate(elems):
                    # U_222[local_idx][s] gives the coefficient for this element in scalar product s
                    U_444[elem][col] += coeff * U_222[local_idx][s]
            
            # For B factors
            for block_idx, coeff in structure['b']:
                elems = block_elements(block_idx)
                for local_idx, elem in enumerate(elems):
                    V_444[elem][col] += coeff * V_222[local_idx][s]
            
            # For C factors
            for block_idx, coeff in structure['c']:
                elems = block_elements(block_idx)
                for local_idx, elem in enumerate(elems):
                    W_444[elem][col] += coeff * W_222[local_idx][s]
    
    # Now build the 4x4x8 decomposition
    # Split B into left half (columns 0-3) and right half (columns 4-7)
    # Left half: B_left is 4x4, right half: B_right is 4x4
    # C_left = A * B_left (4x4), C_right = A * B_right (4x4)
    
    # U has shape (16, 98)
    # V has shape (32, 98): V[0:16] for left half, V[16:32] for right half
    # W has shape (32, 98): W[0:16] for left half, W[16:32] for right half
    
    U = [[0]*98 for _ in range(16)]
    V = [[0]*98 for _ in range(32)]
    W = [[0]*98 for _ in range(32)]
    
    # For the left half (columns 0-3 of B, rows 0-3 of C):
    # Use U_444, V_444, W_444 for columns 0-48 of the rank
    for i in range(16):
        for col in range(49):
            U[i][col] = U_444[i][col]
            V[i][col] = V_444[i][col]  # B_left elements 0-15
            W[i][col] = W_444[i][col]  # C_left elements 0-15
    
    # For the right half (columns 4-7 of B, rows 4-7 of C):
    # Use U_444, V_444, W_444 for columns 49-97 of the rank
    # B_right element (j, k) where j=0..3, k=0..3 corresponds to B element (j, 4+k)
    # C_right element (k, i) where k=0..3, i=0..3 corresponds to C element (4+k, i)
    for i in range(16):
        for col in range(49):
            U[i][col + 49] = U_444[i][col]
            # V_444[i] corresponds to B element index i, which is (j, k) where i = j*4+k
            # For B_right, this element is at position (j, 4+k) in the full B, index j*8 + (4+k)
            j = i // 4
            k = i % 4
            V[i + 16][col + 49] = V_444[i][col]
            # W_444[i] corresponds to C element index i, which is (k, l) where i = k*4+l
            # For C_right, this element is at position (4+k, l) in the full C, index (4+k)*4 + l
            k_w = i // 4
            l_w = i % 4
            W[i + 16][col + 49] = W_444[i][col]
    
    return {"U": U, "V": V, "W": W}
