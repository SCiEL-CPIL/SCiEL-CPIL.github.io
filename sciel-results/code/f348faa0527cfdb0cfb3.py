"""Program for matrix multiplication tensor decomposition (4x5 * 5x5)."""

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_rank7():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indices:
    A (2x2): 00=0, 01=1, 10=2, 11=3
    B (2x2): 00=0, 01=1, 10=2, 11=3
    C (2x2): 00=0, 01=1, 10=2, 11=3
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Product 1: (A00+A11)(B00+B11)
    U[0, 0] += 1
    U[3, 0] += 1
    V[0, 0] += 1
    V[3, 0] += 1
    # C00 += M1, C01 -= M1, C10 += M1, C11 -= M1
    W[0, 0] += 1
    W[1, 0] -= 1
    W[2, 0] += 1
    W[3, 0] -= 1

    # Product 2: (A10+A11)B00
    U[2, 1] += 1
    U[3, 1] += 1
    V[0, 1] += 1
    # C10 += M2
    W[2, 1] += 1

    # Product 3: A00(B01-B11)
    U[0, 2] += 1
    V[1, 2] += 1
    V[3, 2] -= 1
    # C01 += M3
    W[1, 2] += 1

    # Product 4: A11(B10-B00)
    U[3, 3] += 1
    V[2, 3] += 1
    V[0, 3] -= 1
    # C11 += M4
    W[3, 3] += 1

    # Product 5: (A00+A01)B11
    U[0, 4] += 1
    U[1, 4] += 1
    V[3, 4] += 1
    # C00 += M5, C01 += M5
    W[0, 4] += 1
    W[1, 4] += 1

    # Product 6: (A10-A00)(B00+B01)
    U[2, 5] += 1
    U[0, 5] -= 1
    V[0, 5] += 1
    V[1, 5] += 1
    # C10 += M6
    W[2, 5] += 1

    # Product 7: (A01-A11)(B10+B11)
    U[1, 6] += 1
    U[3, 6] -= 1
    V[2, 6] += 1
    V[3, 6] += 1
    # C11 += M7
    W[3, 6] += 1

    return U, V, W

def construct():
    """
    Construct a rank-85 decomposition for 4x5 * 5x5 matrix multiplication.
    
    The decomposition splits the multiplication into:
    1. 4x4 * 4x4 (top-left block): rank 49 via Strassen
    2. 4x4 * 4x1 (left part times right column): rank 16 via schoolbook
    3. 4x1 * 1x5 (right column times top row): rank 20 via schoolbook
    
    Total rank: 49 + 16 + 20 = 85
    """
    n, m, p = 4, 5, 5
    
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    total_rank = 85
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    col = 0
    
    # ============================================================
    # PART 1: 4x4 * 4x4 multiplication using Strassen (rank 49)
    # This handles C[i,k] for i in 0..3, k in 0..3, from sum over j=0..3
    # ============================================================
    
    # Get the 2x2 Strassen factors
    U2, V2, W2 = _strassen_2x2_rank7()
    
    # Define the 2x2 block indices for a 4x4 matrix.
    # A 4x4 matrix is split into four 2x2 blocks:
    #   [A11 | A12]
    #   [A21 | A22]
    # where:
    #   A11 = rows 0-1, cols 0-1
    #   A12 = rows 0-1, cols 2-3
    #   A21 = rows 2-3, cols 0-1
    #   A22 = rows 2-3, cols 2-3
    
    # For a 4x4 matrix with row-major indexing (row * 4 + col):
    # A11: indices 0, 1, 4, 5  (i.e., (0,0),(0,1),(1,0),(1,1))
    # A12: indices 2, 3, 6, 7  (i.e., (0,2),(0,3),(1,2),(1,3))
    # A21: indices 8, 9, 12, 13  (i.e., (2,0),(2,1),(3,0),(3,1))
    # A22: indices 10, 11, 14, 15  (i.e., (2,2),(2,3),(3,2),(3,3))
    
    # The 7 Strassen block products for 4x4 * 4x4:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C block reconstruction:
    # C11 = M1 - M3 + M4 - M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M5 + M7
    
    # For each of the 7 block products, we apply the 2x2 Strassen decomposition.
    # Each block product Mi is a 2x2 * 2x2 multiplication, which requires rank 7.
    # So we need 7 * 7 = 49 scalar multiplications total.
    
    # Define the block index mappings.
    # For matrix A (4x4), flattened as i*4 + j:
    # Block 11 (top-left): local indices 0,1,4,5 -> global indices 0,1,4,5
    # Block 12 (top-right): local indices 0,1,4,5 -> global indices 2,3,6,7
    # Block 21 (bottom-left): local indices 0,1,4,5 -> global indices 8,9,12,13
    # Block 22 (bottom-right): local indices 0,1,4,5 -> global indices 10,11,14,15
    
    # We'll define the 7 block products as linear combinations of the 4 blocks.
    # For each product, we specify:
    #   - Which A blocks contribute and with what coefficients
    #   - Which B blocks contribute and with what coefficients
    
    # Product definitions (coefficients for each of the 4 blocks):
    # A blocks: [11, 12, 21, 22]
    # B blocks: [11, 12, 21, 22]
    
    # M1 = (A11 + A22)(B11 + B22)
    #   A: [1, 0, 0, 1], B: [1, 0, 0, 1]
    # M2 = (A21 + A22)B11
    #   A: [0, 0, 1, 1], B: [1, 0, 0, 0]
    # M3 = A11(B12 - B22)
    #   A: [1, 0, 0, 0], B: [0, 1, 0, -1]
    # M4 = A22(B21 - B11)
    #   A: [0, 0, 0, 1], B: [-1, 0, 1, 0]
    # M5 = (A11 + A12)B22
    #   A: [1, 1, 0, 0], B: [0, 0, 0, 1]
    # M6 = (A21 - A11)(B11 + B12)
    #   A: [-1, 0, 1, 0], B: [1, 1, 0, 0]
    # M7 = (A12 - A22)(B21 + B22)
    #   A: [0, 1, 0, -1], B: [0, 0, 1, 1]
    
    block_products = [
        # (A_coeffs, B_coeffs) where coeffs are [11, 12, 21, 22]
        ([1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1]),  # M7
    ]
    
    # C block reconstruction coefficients:
    # C11 = M1 - M3 + M4 - M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M2 + M5 + M7
    c_block_coeffs = {
        '11': [1, 0, -1, 1, 0, 0, -1],   # C11
        '12': [0, 0, 1, 0, 1, 0, 0],     # C12
        '21': [0, 1, 0, 0, 0, 1, 0],     # C21
        '22': [1, -1, 0, 0, 1, 0, 1],    # C22
    }
    
    # Global index mappings for A (4x4, flattened as i*4 + j):
    # Block 11: local 2x2 indices [0,1,2,3] -> global [0,1,4,5]
    # Block 12: local 2x2 indices [0,1,2,3] -> global [2,3,6,7]
    # Block 21: local 2x2 indices [0,1,2,3] -> global [8,9,12,13]
    # Block 22: local 2x2 indices [0,1,2,3] -> global [10,11,14,15]
    
    a_block_global = {
        '11': [0, 1, 4, 5],
        '12': [2, 3, 6, 7],
        '21': [8, 9, 12, 13],
        '22': [10, 11, 14, 15],
    }
    
    # Global index mappings for B (4x4 part of 5x5, flattened as j*5 + k):
    # B[:4, :4] where B is 5x5, so B[j, k] -> j*5 + k
    # Block 11: rows 0-1, cols 0-1 -> global [0,1,5,6]
    # Block 12: rows 0-1, cols 2-3 -> global [2,3,7,8]
    # Block 21: rows 2-3, cols 0-1 -> global [10,11,15,16]
    # Block 22: rows 2-3, cols 2-3 -> global [12,13,17,18]
    
    b_block_global = {
        '11': [0, 1, 5, 6],
        '12': [2, 3, 7, 8],
        '21': [10, 11, 15, 16],
        '22': [12, 13, 17, 18],
    }
    
    # Global index mappings for C (4x4 part, flattened as k*4 + i per trace convention):
    # C[i, k] -> k*4 + i
    # Block 11: i in 0-1, k in 0-1 -> global [0,1,4,5]
    # Block 12: i in 0-1, k in 2-3 -> global [2,3,6,7]
    # Block 21: i in 2-3, k in 0-1 -> global [8,9,12,13]
    # Block 22: i in 2-3, k in 2-3 -> global [10,11,14,15]
    
    c_block_global = {
        '11': [0, 1, 4, 5],
        '12': [2, 3, 6, 7],
        '21': [8, 9, 12, 13],
        '22': [10, 11, 14, 15],
    }
    
    # Now build the 49 columns for the 4x4*4x4 part.
    for prod_idx, (a_coeffs, b_coeffs) in enumerate(block_products):
        # a_coeffs and b_coeffs are arrays of 4 elements for blocks [11, 12, 21, 22]
        block_names = ['11', '12', '21', '22']
        
        # For each of the 7 Strassen scalar products within this block product:
        for s in range(7):
            # U2[local_a_idx, s] gives the coefficient for A's local element
            # V2[local_b_idx, s] gives the coefficient for B's local element
            # W2[local_c_idx, s] gives the coefficient for C's local element
            
            # For U: sum over A blocks
            for local_a in range(4):
                coeff_a = U2[local_a, s]
                if coeff_a == 0:
                    continue
                # For each A block that contributes to this product:
                for b_idx, block_name in enumerate(block_names):
                    block_coeff = a_coeffs[b_idx]
                    if block_coeff == 0:
                        continue
                    # The local index local_a maps to a global index in this block
                    global_a_idx = a_block_global[block_name][local_a]
                    U[global_a_idx, col + s] += coeff_a * block_coeff
            
            # For V: sum over B blocks
            for local_b in range(4):
                coeff_b = V2[local_b, s]
                if coeff_b == 0:
                    continue
                for b_idx, block_name in enumerate(block_names):
                    block_coeff = b_coeffs[b_idx]
                    if block_coeff == 0:
                        continue
                    global_b_idx = b_block_global[block_name][local_b]
                    V[global_b_idx, col + s] += coeff_b * block_coeff
            
            # For W: sum over C blocks
            for local_c in range(4):
                coeff_c = W2[local_c, s]
                if coeff_c == 0:
                    continue
                # C block reconstruction: C_block = sum over products
                for c_block_name, c_prod_coeffs in c_block_coeffs.items():
                    c_coeff = c_prod_coeffs[prod_idx]
                    if c_coeff == 0:
                        continue
                    global_c_idx = c_block_global[c_block_name][local_c]
                    W[global_c_idx, col + s] += coeff_c * c_coeff
        
        col += 7
    
    # col should be 49 now
    
    # ============================================================
    # PART 2: 4x4 * 4x1 multiplication (rank 16, schoolbook)
    # This handles C[i, 4] for i in 0..3, from sum over j=0..3 of A[i,j]*B[j,4]
    # ============================================================
    
    for i in range(4):
        for j in range(4):
            # A[i, j] -> i*5 + j (since A is 4x5)
            a_idx = i * 5 + j
            # B[j, 4] -> j*5 + 4
            b_idx = j * 5 + 4
            # C[i, 4] -> 4*4 + i = 16 + i
            c_idx = 4 * 4 + i
            
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # col should be 49 + 16 = 65 now
    
    # ============================================================
    # PART 3: 4x1 * 1x5 multiplication (rank 20, schoolbook)
    # This handles C[i, k] for i in 0..3, k in 0..4, from A[i,4]*B[4,k]
    # ============================================================
    
    for i in range(4):
        for k in range(5):
            # A[i, 4] -> i*5 + 4
            a_idx = i * 5 + 4
            # B[4, k] -> 4*5 + k = 20 + k
            b_idx = 4 * 5 + k
            # C[i, k] -> k*4 + i
            c_idx = k * 4 + i
            
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # col should be 65 + 20 = 85 now
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
