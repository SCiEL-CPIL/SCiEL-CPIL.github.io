"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2_factors():
    """Return (U, V, W) for 2x2x2 matrix mult with rank 7 (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M2 = (A10+A11)B00 -> C10+=1
    # M3 = A00(B01-B11) -> C01+=1, C11+=-1
    # M4 = A11(B10-B00) -> C10+=1, C00+=-1
    # M5 = (A00+A01)B11 -> C01+=1, C00+=-1
    # M6 = (A10-A00)(B00+B01) -> C11+=1
    # M7 = (A01-A11)(B10+B11) -> C00+=1, C10+=-1
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices (W): C00=0, C01=1, C10=2, C11=3  (W[k*2+i])
    
    # M1
    U[0,0] = 1; U[3,0] = 1  # A00, A11
    V[0,0] = 1; V[3,0] = 1  # B00, B11
    W[0,0] = 1; W[3,0] = 1  # C00, C11
    
    # M2
    U[2,1] = 1; U[3,1] = 1  # A10, A11
    V[0,1] = 1  # B00
    W[2,1] = 1  # C10
    
    # M3
    U[0,2] = 1  # A00
    V[1,2] = 1; V[3,2] = -1  # B01, -B11
    W[1,2] = 1; W[3,2] = -1  # C01, -C11
    
    # M4
    U[3,3] = 1  # A11
    V[2,3] = 1; V[0,3] = -1  # B10, -B00
    W[2,3] = 1; W[0,3] = -1  # C10, -C00
    
    # M5
    U[0,4] = 1; U[1,4] = 1  # A00, A01
    V[3,4] = 1  # B11
    W[1,4] = 1; W[0,4] = -1  # C01, -C00
    
    # M6
    U[2,5] = 1; U[0,5] = -1  # A10, -A00
    V[0,5] = 1; V[1,5] = 1  # B00, B01
    W[3,5] = 1  # C11
    
    # M7
    U[1,6] = 1; U[3,6] = -1  # A01, -A11
    V[2,6] = 1; V[3,6] = 1  # B10, B11
    W[0,6] = 1; W[2,6] = -1  # C00, -C10
    
    return U, V, W

def _strassen_4x4_4x4_factors():
    """Return (U, V, W) for 4x4x4 matrix mult with rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    Uses recursive Strassen.
    """
    U2, V2, W2 = _strassen_2x2_factors()
    
    # Top-level Strassen for 4x4:
    # Split A into [[A11, A12], [A21, A22]] (each 2x2)
    # Split B into [[B11, B12], [B21, B22]] (each 2x2)
    # Split C into [[C11, C12], [C21, C22]] (each 2x2)
    
    # Top-level Strassen terms (each is a 2x2 matrix mult):
    # M1 = (A11+A22)(B11+B22) -> C11 +=, C22 +=
    # M2 = (A21+A22)B11 -> C21 +=
    # M3 = A11(B12-B22) -> C12 +=, C22 -=
    # M4 = A22(B21-B11) -> C21 +=, C11 -=
    # M5 = (A11+A12)B22 -> C12 +=, C11 -=
    # M6 = (A21-A11)(B11+B12) -> C22 +=
    # M7 = (A12-A22)(B21+B22) -> C11 +=, C21 -=
    
    # For each top-level term, the "A matrix" and "B matrix" are 2x2 blocks,
    # and the result is a 2x2 block of C.
    # Each top-level term involves a 2x2x2 multiplication, which has rank 7.
    
    # Block indices:
    # A blocks: A11 rows 0-1,cols 0-1; A12 rows 0-1,cols 2-3; A21 rows 2-3,cols 0-1; A22 rows 2-3,cols 2-3
    # In U indexing (i*4+j): A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
    # B blocks: B11 rows 0-1,cols 0-1; B12 rows 0-1,cols 2-3; B21 rows 2-3,cols 0-1; B22 rows 2-3,cols 2-3
    # In V indexing (j*4+k): B11: 0-3, B12: 4-7, B21: 8-11, B22: 12-15
    # C blocks: C11 rows 0-1,cols 0-1; C12 rows 0-1,cols 2-3; C21 rows 2-3,cols 0-1; C22 rows 2-3,cols 2-3
    # In W indexing (k*4+i): C11: 0-3, C12: 4-7, C21: 8-11, C22: 12-15
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # For each of the 7 top-level Strassen terms, we have a 2x2x2 sub-problem.
    # The sub-problem's U2, V2, W2 have shape (4,7).
    # We need to map the sub-problem indices to the full 4x4 indices.
    
    # Top-level term definitions:
    # (A_blocks_with_signs, B_blocks_with_signs, C_blocks_with_signs)
    # Each block is a 2x2 matrix.
    
    terms = [
        # M1: (A11+A22)(B11+B22) -> C11+=, C22+=
        {
            'A': [(0, 1), (12, 1)],   # A11 (offset 0), A22 (offset 12), both +1
            'B': [(0, 1), (12, 1)],   # B11 (offset 0), B22 (offset 12), both +1
            'C': [(0, 1), (12, 1)]    # C11 (offset 0), C22 (offset 12), both +1
        },
        # M2: (A21+A22)B11 -> C21+=
        {
            'A': [(8, 1), (12, 1)],   # A21 (offset 8), A22 (offset 12), both +1
            'B': [(0, 1)],            # B11 (offset 0), +1
            'C': [(8, 1)]             # C21 (offset 8), +1
        },
        # M3: A11(B12-B22) -> C12+=, C22-=
        {
            'A': [(0, 1)],            # A11 (offset 0), +1
            'B': [(4, 1), (12, -1)],  # B12 (offset 4) +1, B22 (offset 12) -1
            'C': [(4, 1), (12, -1)]   # C12 (offset 4) +1, C22 (offset 12) -1
        },
        # M4: A22(B21-B11) -> C21+=, C11-=
        {
            'A': [(12, 1)],           # A22 (offset 12), +1
            'B': [(8, 1), (0, -1)],   # B21 (offset 8) +1, B11 (offset 0) -1
            'C': [(8, 1), (0, -1)]    # C21 (offset 8) +1, C11 (offset 0) -1
        },
        # M5: (A11+A12)B22 -> C12+=, C11-=
        {
            'A': [(0, 1), (4, 1)],    # A11 (offset 0), A12 (offset 4), both +1
            'B': [(12, 1)],           # B22 (offset 12), +1
            'C': [(4, 1), (0, -1)]    # C12 (offset 4) +1, C11 (offset 0) -1
        },
        # M6: (A21-A11)(B11+B12) -> C22+=
        {
            'A': [(8, 1), (0, -1)],   # A21 (offset 8) +1, A11 (offset 0) -1
            'B': [(0, 1), (4, 1)],    # B11 (offset 0) +1, B12 (offset 4) +1
            'C': [(12, 1)]            # C22 (offset 12) +1
        },
        # M7: (A12-A22)(B21+B22) -> C11+=, C21-=
        {
            'A': [(4, 1), (12, -1)],  # A12 (offset 4) +1, A22 (offset 12) -1
            'B': [(8, 1), (12, 1)],   # B21 (offset 8) +1, B22 (offset 12) +1
            'C': [(0, 1), (8, -1)]    # C11 (offset 0) +1, C21 (offset 8) -1
        },
    ]
    
    for t_idx, term in enumerate(terms):
        base = t_idx * 7
        for s_idx in range(7):
            col = base + s_idx
            # For the sub-problem, U2[:, s_idx] gives coefficients for the 4 A-entries
            # V2[:, s_idx] gives coefficients for the 4 B-entries
            # W2[:, s_idx] gives coefficients for the 4 C-entries
            
            # Map A sub-indices to full indices
            for sub_i in range(4):
                coeff = U2[sub_i, s_idx]
                if coeff != 0:
                    for (block_offset, sign) in term['A']:
                        # sub_i corresponds to a specific entry in the 2x2 block
                        # A block entries: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1) within block
                        # Full index: block_offset + sub_i
                        U[block_offset + sub_i, col] += coeff * sign
            
            # Map B sub-indices to full indices
            for sub_j in range(4):
                coeff = V2[sub_j, s_idx]
                if coeff != 0:
                    for (block_offset, sign) in term['B']:
                        V[block_offset + sub_j, col] += coeff * sign
            
            # Map C sub-indices to full indices
            for sub_k in range(4):
                coeff = W2[sub_k, s_idx]
                if coeff != 0:
                    for (block_offset, sign) in term['C']:
                        W[block_offset + sub_k, col] += coeff * sign
    
    return U, V, W

def _strassen_4x4_4x2_factors():
    """Return (U, V, W) for 4x4 x 4x2 matrix mult with rank 28.
    U: (16, 28), V: (8, 28), W: (8, 28)
    
    Split into two 2x2 x 2x2 multiplications for top and bottom halves.
    Actually, let me think...
    
    A is 4x4, B is 4x2, C is 4x2.
    Split A into [[A11, A12], [A21, A22]] (2x2 blocks)
    Split B into [[B1], [B2]] where B1, B2 are 2x2
    Split C into [[C1], [C2]] where C1, C2 are 2x2
    
    C1 = A11*B1 + A12*B2  (2x2 = 2x2*2x2 + 2x2*2x2)
    C2 = A21*B1 + A22*B2  (2x2 = 2x2*2x2 + 2x2*2x2)
    
    For C1: need to compute A11*B1 + A12*B2
    This is a 2x4 * 4x2 = 2x2 multiplication.
    I can use Strassen on a 2x2 x 2x2 problem by padding, or handle directly.
    
    Actually, A11*B1 + A12*B2 can be seen as:
    [A11, A12] (2x4) * [B1; B2] (4x2) = 2x2
    
    Let me define new variables:
    P = A11 + A12, Q = A11 - A12
    R = B1 + B2, S = B1 - B2
    
    A11*B1 + A12*B2 = (P+Q)/2 * (R+S)/2 + (P-Q)/2 * (R-S)/2
    = (1/4)[(P+Q)(R+S) + (P-Q)(R-S)]
    = (1/4)[PR + PS + QR + QS + PR - PS - QR + QS]
    = (1/4)[2PR + 2QS]
    = (1/2)(PR + QS)
    
    So A11*B1 + A12*B2 = (1/2)[(A11+A12)(B1+B2) + (A11-A12)(B1-B2)]
    
    Each of (A11+A12)(B1+B2) and (A11-A12)(B1-B2) is a 2x2*2x2 product, rank 7.
    So C1 needs 14 rank-1 terms (with 1/2 factor).
    
    Similarly, C2 = A21*B1 + A22*B2 = (1/2)[(A21+A22)(B1+B2) + (A21-A22)(B1-B2)]
    C2 needs 14 rank-1 terms.
    
    Total: 28 rank-1 terms. 
    
    But wait, can I share (B1+B2) and (B1-B2) between C1 and C2?
    Yes! The V vectors for (B1+B2) and (B1-B2) are the same for both C1 and C2.
    But the U vectors are different (A11+A12 vs A21+A22, etc.)
    
    So no sharing of the full rank-1 terms, but the structure is clear.
    """
    U2, V2, W2 = _strassen_2x2_factors()
    
    U = np.zeros((16, 28))
    V = np.zeros((8, 28))
    W = np.zeros((8, 28))
    
    # A indices (i*4+j):
    # A11: 0,1,4,5 (rows 0-1, cols 0-1)
    # A12: 2,3,6,7 (rows 0-1, cols 2-3)
    # A21: 8,9,12,13 (rows 2-3, cols 0-1)
    # A22: 10,11,14,15 (rows 2-3, cols 2-3)
    
    # B indices (j*2+k), B is 4x2:
    # B1 (rows 0-1): B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # B2 (rows 2-3): B[2,0]=4, B[2,1]=5, B[3,0]=6, B[3,1]=7
    
    # C indices (k*2+i), C is 4x2:
    # C1 (rows 0-1): C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    # C2 (rows 2-3): C[2,0]=4, C[2,1]=5, C[3,0]=6, C[3,1]=7
    
    # A11+A12: entries 0,1,2,3,4,5,6,7 all +1
    # A11-A12: entries 0,1,4,5 +1; entries 2,3,6,7 -1
    # A21+A22: entries 8,9,10,11,12,13,14,15 all +1
    # A21-A22: entries 8,9,12,13 +1; entries 10,11,14,15 -1
    
    # B1+B2: all 8 entries +1
    # B1-B2: entries 0-3 +1, entries 4-7 -1
    
    # C1: W entries 0-3
    # C2: W entries 4-7
    
    # For C1 = (1/2)[(A11+A12)(B1+B2) + (A11-A12)(B1-B2)]
    # Term 1: (A11+A12)(B1+B2) -> C1, coefficient 1/2
    # Term 2: (A11-A12)(B1-B2) -> C1, coefficient 1/2
    
    # For C2 = (1/2)[(A21+A22)(B1+B2) + (A21-A22)(B1-B2)]
    # Term 3: (A21+A22)(B1+B2) -> C2, coefficient 1/2
    # Term 4: (A21-A22)(B1-B2) -> C2, coefficient 1/2
    
    # Each term is a 2x2*2x2 product, using Strassen (7 rank-1 terms)
    
    def _add_strassen_term(U_full, V_full, W_full, base_col, 
                           A_indices, A_signs, B_indices, B_signs, C_indices, C_signs,
                           U2, V2, W2, scale=1.0):
        """Add 7 Strassen rank-1 terms starting at base_col."""
        for s in range(7):
            col = base_col + s
            for si in range(4):
                coeff = U2[si, s] * scale
                if coeff != 0:
                    idx = A_indices[si]
                    U_full[idx, col] += coeff * A_signs[si]
            for sj in range(4):
                coeff = V2[sj, s] * scale
                if coeff != 0:
                    idx = B_indices[sj]
                    V_full[idx, col] += coeff * B_signs[sj]
            for sk in range(4):
                coeff = W2[sk, s] * scale
                if coeff != 0:
                    idx = C_indices[sk]
                    W_full[idx, col] += coeff * C_signs[sk]
    
    # A11+A12: all 8 entries of A in rows 0-1
    # The 2x2 "matrix" is formed by A11 and A12 side by side
    # But for Strassen, I need to treat (A11+A12) as a 2x2 matrix
    # A11+A12 = [[A11_00+A12_00, A11_01+A12_01], [A11_10+A12_10, A11_11+A12_11]]
    # In U indexing: A11_00=0, A11_01=1, A11_10=4, A11_11=5
    #                A12_00=2, A12_01=3, A12_10=6, A12_11=7
    # So (A11+A12) sub-matrix entries: [0+2, 1+3, 4+6, 5+7] = [0,1,4,5] for A11 and [2,3,6,7] for A12
    # Wait, I need to map the 2x2 sub-matrix indices to full A indices.
    # Sub-matrix entry (0,0) = A11_00 + A12_00 -> U[0] and U[2]
    # Sub-matrix entry (0,1) = A11_01 + A12_01 -> U[1] and U[3]
    # Sub-matrix entry (1,0) = A11_10 + A12_10 -> U[4] and U[6]
    # Sub-matrix entry (1,1) = A11_11 + A12_11 -> U[5] and U[7]
    
    # For the Strassen sub-problem, U2[si, s] gives the coefficient for sub-entry si.
    # I need to distribute this to the corresponding full A indices.
    
    # Let me define the mapping more carefully.
    # For (A11+A12) as a 2x2 matrix:
    #   sub(0,0) -> full indices [0, 2] (A11_00, A12_00), both +1
    #   sub(0,1) -> full indices [1, 3] (A11_01, A12_01), both +1
    #   sub(1,0) -> full indices [4, 6] (A11_10, A12_10), both +1
    #   sub(1,1) -> full indices [5, 7] (A11_11, A12_11), both +1
    
    # For (A11-A12):
    #   sub(0,0) -> [0, 2] with signs [+1, -1]
    #   sub(0,1) -> [1, 3] with signs [+1, -1]
    #   sub(1,0) -> [4, 6] with signs [+1, -1]
    #   sub(1,1) -> [5, 7] with signs [+1, -1]
    
    # For (A21+A22):
    #   sub(0,0) -> [8, 10] (A21_00, A22_00), both +1
    #   sub(0,1) -> [9, 11] (A21_01, A22_01), both +1
    #   sub(1,0) -> [12, 14] (A21_10, A22_10), both +1
    #   sub(1,1) -> [13, 15] (A21_11, A22_11), both +1
    
    # For (A21-A22):
    #   sub(0,0) -> [8, 10] with signs [+1, -1]
    #   sub(0,1) -> [9, 11] with signs [+1, -1]
    #   sub(1,0) -> [12, 14] with signs [+1, -1]
    #   sub(1,1) -> [13, 15] with signs [+1, -1]
    
    # For (B1+B2) as a 2x2 matrix:
    #   sub(0,0) -> [0, 4] (B1_00, B2_00), both +1
    #   sub(0,1) -> [1, 5] (B1_01, B2_01), both +1
    #   sub(1,0) -> [2, 6] (B1_10, B2_10), both +1
    #   sub(1,1) -> [3, 7] (B1_11, B2_11), both +1
    
    # For (B1-B2):
    #   sub(0,0) -> [0, 4] with signs [+1, -1]
    #   sub(0,1) -> [1, 5] with signs [+1, -1]
    #   sub(1,0) -> [2, 6] with signs [+1, -1]
    #   sub(1,1) -> [3, 7] with signs [+1, -1]
    
    # For C1 (2x2):
    #   sub(0,0) -> [0], sub(0,1) -> [1], sub(1,0) -> [2], sub(1,1) -> [3]
    
    # For C2 (2x2):
    #   sub(0,0) -> [4], sub(0,1) -> [5], sub(1,0) -> [6], sub(1,1) -> [7]
    
    def _add_term(U_full, V_full, W_full, base_col, 
                  A_map, B_map, C_map, scale=1.0):
        """A_map: list of 4 tuples (full_idx_list, sign_list) for each sub-entry
           B_map: same for B
           C_map: same for C
        """
        for s in range(7):
            col = base_col + s
            for si in range(4):
                coeff = U2[si, s] * scale
                if coeff != 0:
                    for (fidx, sgn) in A_map[si]:
                        U_full[fidx, col] += coeff * sgn
            for sj in range(4):
                coeff = V2[sj, s] * scale
                if coeff != 0:
                    for (fidx, sgn) in B_map[sj]:
                        V_full[fidx, col] += coeff * sgn
            for sk in range(4):
                coeff = W2[sk, s] * scale
                if coeff != 0:
                    for (fidx, sgn) in C_map[sk]:
                        W_full[fidx, col] += coeff * sgn
    
    # A11+A12 map
    A_plus_11_12 = [
        [(0, 1), (2, 1)],   # sub(0,0)
        [(1, 1), (3, 1)],   # sub(0,1)
        [(4, 1), (6, 1)],   # sub(1,0)
        [(5, 1), (7, 1)],   # sub(1,1)
    ]
    # A11-A12 map
    A_minus_11_12 = [
        [(0, 1), (2, -1)],
        [(1, 1), (3, -1)],
        [(4, 1), (6, -1)],
        [(5, 1), (7, -1)],
    ]
    # A21+A22 map
    A_plus_21_22 = [
        [(8, 1), (10, 1)],
        [(9, 1), (11, 1)],
        [(12, 1), (14, 1)],
        [(13, 1), (15, 1)],
    ]
    # A21-A22 map
    A_minus_21_22 = [
        [(8, 1), (10, -1)],
        [(9, 1), (11, -1)],
        [(12, 1), (14, -1)],
        [(13, 1), (15, -1)],
    ]
    # B1+B2 map
    B_plus = [
        [(0, 1), (4, 1)],
        [(1, 1), (5, 1)],
        [(2, 1), (6, 1)],
        [(3, 1), (7, 1)],
    ]
    # B1-B2 map
    B_minus = [
        [(0, 1), (4, -1)],
        [(1, 1), (5, -1)],
        [(2, 1), (6, -1)],
        [(3, 1), (7, -1)],
    ]
    # C1 map
    C1_map = [
        [(0, 1)],
        [(1, 1)],
        [(2, 1)],
        [(3, 1)],
    ]
    # C2 map
    C2_map = [
        [(4, 1)],
        [(5, 1)],
        [(6, 1)],
        [(7, 1)],
    ]
    
    # C1 = (1/2)[(A11+A12)(B1+B2) + (A11-A12)(B1-B2)]
    _add_term(U, V, W, 0, A_plus_11_12, B_plus, C1_map, scale=0.5)
    _add_term(U, V, W, 7, A_minus_11_12, B_minus, C1_map, scale=0.5)
    
    # C2 = (1/2)[(A21+A22)(B1+B2) + (A21-A22)(B1-B2)]
    _add_term(U, V, W, 14, A_plus_21_22, B_plus, C2_map, scale=0.5)
    _add_term(U, V, W, 21, A_minus_21_22, B_minus, C2_map, scale=0.5)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    U4x4, V4x4, W4x4 = _strassen_4x4_4x4_factors()
    U4x2, V4x2, W4x2 = _strassen_4x4_4x2_factors()
    
    rank = 49 + 28  # 77
    
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # 4x4x4 part: first 49 columns
    # U: (16, 49) -> U[0:16, 0:49]
    # V: (16, 49) -> V[0:16, 0:49]  (B is 4x4, indices 0-15)
    # W: (16, 49) -> W[0:16, 0:49]  (C is 4x4, indices 0-15)
    U[:, 0:49] = U4x4
    V[0:16, 0:49] = V4x4
    W[0:16, 0:49] = W4x4
    
    # 4x4x2 part: last 28 columns
    # U: (16, 28) -> U[0:16, 49:77]
    # V: (8, 28) -> V[16:24, 49:77]  (B rows 2-3, i.e., last 8 of 24)
    # W: (8, 28) -> W[16:24, 49:77]  (C rows 2-3, i.e., last 8 of 24)
    U[:, 49:77] = U4x2
    V[16:24, 49:77] = V4x2
    W[16:24, 49:77] = W4x2
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Safety: return what we have
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
