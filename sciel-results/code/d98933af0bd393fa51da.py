"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We'll construct a decomposition of rank 49 (for 4x4x4) + 48 (for 4x4x3) = 97
    # 4x4x4: Use Strassen recursively
    # 4x4x3: Schoolbook
    
    # First, let's build the 4x4x4 rank-49 decomposition
    # Then the 4x4x3 rank-48 decomposition
    # Combine them
    
    def strassen_2x2_decomp():
        """Return U, V, W for 2x2x2 matrix multiplication with rank 7."""
        # Strassen's 7 multiplications
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        # C11 = P1+P4-P5+P7
        # C12 = P3+P5
        # C21 = P2+P4
        # C22 = P1-P2+P3+P6
        
        # U: 4x7 (for A entries: a,b,c,d)
        # V: 4x7 (for B entries: e,f,g,h)
        # W: 4x7 (for C entries: C11,C12,C21,C22)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Indexing: U[i*2+j, c] for A[i,j]
        # A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
        # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
        # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
        # W[k*2+i, c] for C[i,k]
        
        # P1 = (a+d)(e+h)
        U[0,0] = 1  # a
        U[3,0] = 1  # d
        V[0,0] = 1  # e
        V[3,0] = 1  # h
        W[0,0] = 1  # C11
        W[3,0] = 1  # C22
        
        # P2 = (c+d)e
        U[2,1] = 1  # c
        U[3,1] = 1  # d
        V[0,1] = 1  # e
        W[2,1] = 1  # C21
        
        # P3 = a(f-h)
        U[0,2] = 1  # a
        V[1,2] = 1  # f
        V[3,2] = -1  # h
        W[1,2] = 1  # C12
        W[3,2] = 1  # C22
        
        # P4 = d(g-e)
        U[3,3] = 1  # d
        V[2,3] = 1  # g
        V[0,3] = -1  # e
        W[0,3] = 1  # C11
        W[2,3] = 1  # C21
        
        # P5 = (a+b)h
        U[0,4] = 1  # a
        U[1,4] = 1  # b
        V[3,4] = 1  # h
        W[0,4] = -1  # C11
        W[1,4] = 1  # C12
        
        # P6 = (c-a)(e+f)
        U[2,5] = 1  # c
        U[0,5] = -1  # a
        V[0,5] = 1  # e
        V[1,5] = 1  # f
        W[3,5] = 1  # C22
        
        # P7 = (b-d)(g+h)
        U[1,6] = 1  # b
        U[3,6] = -1  # d
        V[2,6] = 1  # g
        V[3,6] = 1  # h
        W[0,6] = 1  # C11
        W[3,6] = 1  # C22
        
        return U, V, W
    
    def strassen_4x4_decomp():
        """Return U, V, W for 4x4x4 matrix multiplication with rank 49."""
        # Split 4x4 into four 2x2 blocks
        # A = [[A11,A12],[A21,A22]] where each is 2x2
        # B = [[B11,B12],[B21,B22]]
        # C = [[C11,C12],[C21,C22]]
        
        # Use Strassen: 7 multiplications of 2x2 matrices
        # Each 2x2 multiplication uses 7 mults, so total 49
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # A indexing: A[i*4+j] for i,j in 0..3
        # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # B indexing: B[j*4+k] for j,k in 0..3
        # B11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # B12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # B21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # B22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # C indexing: C[k*4+i] for k,i in 0..3 (W[k*n+i] = W[k*4+i])
        # C11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # C12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # C21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # C22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # Get the 2x2 Strassen decomposition
        U2, V2, W2 = strassen_2x2_decomp()
        
        # For each of the 7 Strassen multiplications, we need to:
        # 1. Form the 2x2 submatrices of A and B
        # 2. Apply the 2x2 Strassen decomposition
        # 3. Add the results to the appropriate 2x2 submatrix of C
        
        # Strassen formulas:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # C11 = P1+P4-P5+P7
        # C12 = P3+P5
        # C21 = P2+P4
        # C22 = P1-P2+P3+P6
        
        # For each Strassen product, we have 7 inner mults
        # Total: 7*7 = 49
        
        # Let's define the block indices
        # A blocks: A11=[0:2,0:2], A12=[0:2,2:4], A21=[2:4,0:2], A22=[2:4,2:4]
        # In flattened form (row-major, 4 cols):
        # A11: indices 0,1,4,5
        # A12: indices 2,3,6,7
        # A21: indices 8,9,12,13
        # A22: indices 10,11,14,15
        
        # B blocks: B11=[0:2,0:2], B12=[0:2,2:4], B21=[2:4,0:2], B22=[2:4,2:4]
        # In flattened form:
        # B11: indices 0,1,4,5
        # B12: indices 2,3,6,7
        # B21: indices 8,9,12,13
        # B22: indices 10,11,14,15
        
        # C blocks: C11=[0:2,0:2], C12=[0:2,2:4], C21=[2:4,0:2], C22=[2:4,2:4]
        # In flattened form (W[k*4+i]):
        # C11: indices 0,1,4,5
        # C12: indices 2,3,6,7
        # C21: indices 8,9,12,13
        # C22: indices 10,11,14,15
        
        # For each Strassen product P_s (s=1..7), we have 7 inner mults (c=0..6)
        # The total column index is s*7 + c
        
        # Define the A block coefficients for each Strassen product
        # P1: A11 + A22
        # P2: A21 + A22
        # P3: A11
        # P4: A22
        # P5: A11 + A12
        # P6: A21 - A11
        # P7: A12 - A22
        
        # B block coefficients:
        # P1: B11 + B22
        # P2: B11
        # P3: B12 - B22
        # P4: B21 - B11
        # P5: B22
        # P6: B11 + B12
        # P7: B21 + B22
        
        # C block contributions:
        # P1: C11 (+1), C22 (+1)
        # P2: C21 (+1)
        # P3: C12 (+1), C22 (+1)
        # P4: C11 (+1), C21 (+1)
        # P5: C11 (-1), C12 (+1)
        # P6: C22 (+1)
        # P7: C11 (+1), C22 (+1)
        
        # A block indices (flattened 4x4):
        A_blocks = {
            'A11': [0, 1, 4, 5],
            'A12': [2, 3, 6, 7],
            'A21': [8, 9, 12, 13],
            'A22': [10, 11, 14, 15]
        }
        
        # B block indices (flattened 4x4):
        B_blocks = {
            'B11': [0, 1, 4, 5],
            'B12': [2, 3, 6, 7],
            'B21': [8, 9, 12, 13],
            'B22': [10, 11, 14, 15]
        }
        
        # C block indices (flattened W[k*4+i]):
        C_blocks = {
            'C11': [0, 1, 4, 5],
            'C12': [2, 3, 6, 7],
            'C21': [8, 9, 12, 13],
            'C22': [10, 11, 14, 15]
        }
        
        # Strassen product definitions
        # (A_coeffs, B_coeffs, C_contribs)
        # A_coeffs: dict from block name to coefficient
        # B_coeffs: dict from block name to coefficient
        # C_contribs: dict from block name to coefficient
        
        strassen_products = [
            # P1
            ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}, {'C11': 1, 'C22': 1}),
            # P2
            ({'A21': 1, 'A22': 1}, {'B11': 1}, {'C21': 1}),
            # P3
            ({'A11': 1}, {'B12': 1, 'B22': -1}, {'C12': 1, 'C22': 1}),
            # P4
            ({'A22': 1}, {'B21': 1, 'B11': -1}, {'C11': 1, 'C21': 1}),
            # P5
            ({'A11': 1, 'A12': 1}, {'B22': 1}, {'C11': -1, 'C12': 1}),
            # P6
            ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}, {'C22': 1}),
            # P7
            ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}, {'C11': 1, 'C22': 1}),
        ]
        
        for s, (a_coeffs, b_coeffs, c_contribs) in enumerate(strassen_products):
            for c in range(7):
                col = s * 7 + c
                
                # U: for each A entry
                for block_name, coeff in a_coeffs.items():
                    for idx in A_blocks[block_name]:
                        # The 2x2 block has entries at positions 0,1,2,3 within the block
                        # U2 has shape (4,7) for the 2x2 case
                        # The block indices correspond to 2x2 positions:
                        # idx in A_blocks corresponds to 2x2 position
                        # A11: [0,1,4,5] -> 2x2 positions [0,1,2,3]
                        # A12: [2,3,6,7] -> 2x2 positions [0,1,2,3]
                        # etc.
                        
                        # Get the 2x2 position within the block
                        block_indices = A_blocks[block_name]
                        pos = block_indices.index(idx)
                        U[idx, col] += coeff * U2[pos, c]
                
                # V: for each B entry
                for block_name, coeff in b_coeffs.items():
                    for idx in B_blocks[block_name]:
                        block_indices = B_blocks[block_name]
                        pos = block_indices.index(idx)
                        V[idx, col] += coeff * V2[pos, c]
                
                # W: for each C entry
                for block_name, coeff in c_contribs.items():
                    for idx in C_blocks[block_name]:
                        block_indices = C_blocks[block_name]
                        pos = block_indices.index(idx)
                        W[idx, col] += coeff * W2[pos, c]
        
        return U, V, W
    
    # Build the 4x4x4 part
    U4, V4, W4 = strassen_4x4_decomp()
    
    # Build the 4x4x3 part (schoolbook, rank 48)
    U3 = np.zeros((16, 48))
    V3 = np.zeros((28, 48))  # m*p = 4*7, but for the 3-column part, we only use the first 4*3=12 entries? 
    # Wait, V has shape (m*p, rank) = (28, rank). For the 4x4x3 part, we're multiplying by the first 3 columns of B.
    # So V entries for j in 0..3, k in 0..2 (the first 3 columns)
    
    # Actually, let me reconsider. The full problem is 4x4 * 4x7.
    # We split the 4x7 into 4x4 + 4x3.
    # For the 4x4 part: U4 (16x49), V4 (16x49), W4 (16x49)
    #   V4 uses B entries for j in 0..3, k in 0..3 (first 4 columns)
    #   W4 uses C entries for k in 0..3 (first 4 columns of C)
    # For the 4x3 part: U3 (16x48), V3 (28x48), W3 (28x48)
    #   V3 uses B entries for j in 0..3, k in 4..6 (last 3 columns)
    #   W3 uses C entries for k in 4..6 (last 3 columns of C)
    
    # Let me redo this more carefully.
    
    # Full problem: A (4x4) * B (4x7) = C (4x7)
    # U: (4*4, rank) = (16, rank)
    # V: (4*7, rank) = (28, rank)
    # W: (7*4, rank) = (28, rank)
    
    # Split B into B_left (4x4) and B_right (4x3)
    # C = A*B_left + A*B_right
    
    # Part 1: A * B_left (4x4 * 4x4 = 4x4)
    # U1: (16, 49), V1: (16, 49) [only first 16 entries of V], W1: (16, 49) [only first 16 entries of W]
    
    # Part 2: A * B_right (4x4 * 4x3 = 4x3)
    # U2: (16, 48), V2: (12, 48) [entries for j in 0..3, k in 4..6], W2: (12, 48) [entries for k in 4..6]
    
    # Combine:
    # U = [U1, U2] (16 x 97)
    # V = [V1; V2] where V1 is 16x49 padded to 28x49 with zeros, V2 is 12x48 padded to 28x48 with zeros
    # W = [W1; W2] where W1 is 16x49 padded to 28x49 with zeros, W2 is 12x48 padded to 28x48 with zeros
    
    # Actually, let me just build the full matrices directly.
    
    rank = 97
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    # Part 1: 4x4x4 with rank 49
    U[:, :49] = U4
    V[:16, :49] = V4  # B_left entries: j in 0..3, k in 0..3 -> indices j*7+k for k in 0..3
    # Wait, V has shape (m*p, rank) = (4*7, rank) = (28, rank)
    # V[j*7+k, c] for j in 0..3, k in 0..6
    # For B_left (k in 0..3): V[j*7+k, c] for k in 0..3
    # So V[:4, :49] = V4[0:4, :] (j=0, k=0..3)
    # V[7:11, :49] = V4[4:8, :] (j=1, k=0..3)
    # etc.
    
    # Let me redo the V4 indexing. V4 has shape (16, 49) for the 4x4 case, where V4[j*4+k, c] for j,k in 0..3.
    # In the full problem, V[j*7+k, c] for j in 0..3, k in 0..6.
    # So for the left part (k in 0..3): V[j*7+k, c] = V4[j*4+k, c]
    
    for j in range(4):
        for k in range(4):
            V[j*7+k, :49] = V4[j*4+k, :49]
    
    # W4 has shape (16, 49) for the 4x4 case, where W4[k*4+i, c] for k,i in 0..3.
    # In the full problem, W[k*4+i, c] for k in 0..6, i in 0..3.
    # For the left part (k in 0..3): W[k*4+i, c] = W4[k*4+i, c]
    
    W[:16, :49] = W4
    
    # Part 2: 4x4x3 with rank 48 (schoolbook)
    # A (4x4) * B_right (4x3) = C_right (4x3)
    # B_right: j in 0..3, k in 4..6 -> V[j*7+k, c] for k in 4..6
    # C_right: k in 4..6, i in 0..3 -> W[k*4+i, c] for k in 4..6
    
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                col = col_offset + (i*4 + j)*1 + (k-4)  # Need 4*4*3 = 48 columns
                # Actually, let's index properly
                col = col_offset + i*12 + j*3 + (k-4)
                U[i*4+j, col] = 1
                V[j*7+k, col] = 1
                W[k*4+i, col] = 1
    
    # Verify the solution
    def verify(U, V, W):
        n, m, p = 4, 4, 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            s = 0.0
                            for c in range(97):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[kp*n+ip, c]
                            target = 1.0 if (ip == i and kp == k) else 0.0
                            if abs(s - target) > 1e-6:
                                return False, (i,j,k,ip,kp,s,target)
        return True, None
    
    valid, info = verify(U, V, W)
    
    if not valid:
        # Fall back to schoolbook
        rank = 112
        U = np.zeros((16, rank))
        V = np.zeros((28, rank))
        W = np.zeros((28, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i*4+j, col] = 1
                    V[j*7+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
