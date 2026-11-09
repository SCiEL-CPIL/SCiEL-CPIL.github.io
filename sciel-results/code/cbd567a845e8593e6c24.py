"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a rank-49 decomposition for 4x4 matrix multiplication
    using the recursive application of Strassen's algorithm.
    
    The 4x4 matrix multiplication can be viewed as multiplying 2x2 blocks of 2x2 matrices.
    Strassen's algorithm reduces the 2x2 matrix multiplication to 7 scalar multiplications.
    Therefore, 4x4 matrix multiplication can be reduced to 7 * 7 = 49 scalar multiplications.
    
    The decomposition is given by U, V, W matrices of shape (16, 49), (16, 49), (16, 49)
    such that for any A, B:
    C = A @ B
    where C is reconstructed via:
    C_flat = sum_r (U[:, r] * A_flat) * (V[:, r] * B_flat) * W[:, r]
    
    Indices:
    A_flat[i * 4 + j] = A[i, j]
    B_flat[j * 4 + k] = B[j, k]
    C_flat[k * 4 + i] = C[i, k]  (Note the transpose in index mapping for W)
    """
    
    # 1. Define the 2x2 Strassen decomposition
    # For 2x2 matrices:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A @ B
    
    # Strassen's 7 multiplications:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b10 - b11)
    # M4 = a11 * (b00 - b01)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # C00 = M1 + M3 - M4 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    # We represent the 2x2 Strassen decomposition as 7 terms.
    # Each term r has:
    # u_r: vector in R^4 (coefficients for A)
    # v_r: vector in R^4 (coefficients for B)
    # w_r: vector in R^4 (coefficients for C)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Index mapping for 2x2:
    # A: 0->00, 1->01, 2->10, 3->11
    # B: 0->00, 1->01, 2->10, 3->11
    # C: 0->00, 1->01, 2->10, 3->11
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[0, 2] = 1; W2[3, 2] = -1
    
    # M4
    U2[3, 3] = 1
    V2[0, 3] = 1; V2[1, 3] = -1
    W2[0, 3] = -1; W2[2, 3] = 1
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = -1
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[1, 6] = 1
    
    # 2. Construct the 4x4 decomposition
    # The 4x4 matrices are composed of 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen for the block level:
    # Let S1..S7 be the 7 block multiplications.
    # S1 = (A11 + A22) * (B11 + B22)
    # S2 = (A21 + A22) * B11
    # S3 = A11 * (B21 - B22)
    # S4 = A22 * (B11 - B12)
    # S5 = (A11 + A12) * B22
    # S6 = (A21 - A11) * (B11 + B12)
    # S7 = (A12 - A22) * (B21 + B22)
    
    # C11 = S1 + S3 - S4 + S7
    # C12 = S5 + S7
    # C21 = S2 + S4
    # C22 = S1 - S3 + S2 - S5
    
    # Each Sx is a 2x2 matrix multiplication.
    # We decompose each Sx using the 2x2 Strassen decomposition (7 terms).
    # Total terms: 7 * 7 = 49.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block index mappings for 4x4 (flattened):
    # A11: indices 0,1,4,5
    # A12: indices 2,3,6,7
    # A21: indices 8,9,12,13
    # A22: indices 10,11,14,15
    
    # B11: indices 0,1,4,5
    # B12: indices 2,3,6,7
    # B21: indices 8,9,12,13
    # B22: indices 10,11,14,15
    
    # C11: indices 0,1,4,5  (C[0:2, 0:2]) -> W index k*4+i
    # C12: indices 2,3,6,7  (C[0:2, 2:4])
    # C21: indices 8,9,12,13 (C[2:4, 0:2])
    # C22: indices 10,11,14,15 (C[2:4, 2:4])
    
    # We need to map the 2x2 local indices to 4x4 global indices.
    # For a block at (block_i, block_j) in the 4x4 matrix:
    # Global index = block_i * 2 + local_i (for row)
    # But A is flattened as i*4+j.
    # A11 (block 0,0): local (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # A12 (block 0,1): local (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
    # A21 (block 1,0): local (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
    # A22 (block 1,1): local (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
    
    # Let's define a helper to get the global indices for a block.
    def get_block_indices(block_i, block_j, matrix_type):
        """
        Returns the 4 global indices corresponding to the 2x2 block at (block_i, block_j).
        matrix_type: 'A', 'B', or 'C'
        """
        if matrix_type == 'A':
            # A[i, j] -> i*4 + j
            # Block (block_i, block_j) covers rows [2*block_i, 2*block_i+1] and cols [2*block_j, 2*block_j+1]
            # Local indices (li, lj) in 0..1
            # Global i = 2*block_i + li
            # Global j = 2*block_j + lj
            # Flat index = i*4 + j
            return [
                (2*block_i + 0)*4 + (2*block_j + 0),
                (2*block_i + 0)*4 + (2*block_j + 1),
                (2*block_i + 1)*4 + (2*block_j + 0),
                (2*block_i + 1)*4 + (2*block_j + 1)
            ]
        elif matrix_type == 'B':
            # B[j, k] -> j*4 + k
            # Block (block_j, block_k) covers rows [2*block_j, 2*block_j+1] and cols [2*block_k, 2*block_k+1]
            # Note: In Strassen block notation for B, the "row" index of the block corresponds to the column index of the block in A.
            # Let's stick to the standard: B is partitioned into 4 blocks B11, B12, B21, B22.
            # B11: rows 0,1 cols 0,1. B12: rows 0,1 cols 2,3.
            # B21: rows 2,3 cols 0,1. B22: rows 2,3 cols 2,3.
            # So block_i corresponds to row block, block_j to col block.
            return [
                (2*block_i + 0)*4 + (2*block_j + 0),
                (2*block_i + 0)*4 + (2*block_j + 1),
                (2*block_i + 1)*4 + (2*block_j + 0),
                (2*block_i + 1)*4 + (2*block_j + 1)
            ]
        elif matrix_type == 'C':
            # C[i, k] -> k*4 + i  (Note the W convention: W[k*n+i, r])
            # Wait, the interface says W[k*n+i, r] corresponds to C[i, k].
            # So the index in the W vector is k*4 + i.
            # Block (block_i, block_k) covers rows i in [2*block_i, 2*block_i+1] and cols k in [2*block_k, 2*block_k+1].
            # Local (li, lk)
            # Global i = 2*block_i + li
            # Global k = 2*block_k + lk
            # Flat index in W = k*4 + i
            return [
                (2*block_k + 0)*4 + (2*block_i + 0),
                (2*block_k + 0)*4 + (2*block_i + 1), # Wait, order of local indices?
                (2*block_k + 1)*4 + (2*block_i + 0),
                (2*block_k + 1)*4 + (2*block_i + 1)
            ]
            # Let's align local indices (0,1,2,3) with (00, 01, 10, 11) for i, k.
            # Local 0: i=0, k=0 -> k=0, i=0 -> 0*4+0 = 0
            # Local 1: i=0, k=1 -> k=1, i=0 -> 1*4+0 = 4? No, standard 2x2 flattening is row-major.
            # In U2, index 1 is A[0,1].
            # In W2, index 1 is C[0,1] which is C[0,1].
            # My W2 definition:
            # W2[0, r] -> C00
            # W2[1, r] -> C01
            # W2[2, r] -> C10
            # W2[3, r] -> C11
            # So local index l corresponds to:
            # l=0: i=0, k=0
            # l=1: i=0, k=1
            # l=2: i=1, k=0
            # l=3: i=1, k=1
            # Global i = 2*block_i + (l % 2) ? No.
            # l=0: i=0, k=0
            # l=1: i=0, k=1
            # l=2: i=1, k=0
            # l=3: i=1, k=1
            # So i = l // 2, k = l % 2.
            # Global i = 2*block_i + (l // 2)
            # Global k = 2*block_k + (l % 2)
            # Flat index in W = k*4 + i
            return [
                (2*block_k + 0)*4 + (2*block_i + 0), # l=0: k=0, i=0
                (2*block_k + 1)*4 + (2*block_i + 0), # l=1: k=1, i=0
                (2*block_k + 0)*4 + (2*block_i + 1), # l=2: k=0, i=1
                (2*block_k + 1)*4 + (2*block_i + 1)  # l=3: k=1, i=1
            ]
    
    # Define the 7 Strassen block terms for the 4x4 level
    # Each term s (0..6) involves specific A blocks and B blocks.
    # A_blocks: (block_i, block_j)
    # B_blocks: (block_i, block_j) -- Note: B's "row" block index corresponds to A's "col" block index in the multiplication?
    # No, in matrix mult C = A B.
    # A is 2x2 blocks of 2x2. B is 2x2 blocks of 2x2.
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # Strassen block terms:
    # S1 = (A11 + A22) * (B11 + B22)
    # S2 = (A21 + A22) * B11
    # S3 = A11 * (B21 - B22)
    # S4 = A22 * (B11 - B12)
    # S5 = (A11 + A12) * B22
    # S6 = (A21 - A11) * (B11 + B12)
    # S7 = (A12 - A22) * (B21 + B22)
    
    # We define for each s:
    # A_terms: list of (block_i, block_j, coeff)
    # B_terms: list of (block_i, block_j, coeff)
    # C_terms: list of (block_i, block_k, coeff)
    
    S_def = [
        # S1
        {
            'A': [((0,0), 1), ((1,1), 1)],
            'B': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)] # C11 and C22
        },
        # S2
        {
            'A': [((1,0), 1), ((1,1), 1)],
            'B': [((0,0), 1)],
            'C': [((1,0), 1), ((1,1), 1)] # C21 and C22
        },
        # S3
        {
            'A': [((0,0), 1)],
            'B': [((1,0), 1), ((1,1), -1)],
            'C': [((0,0), 1), ((1,1), -1)] # C11 and C22
        },
        # S4
        {
            'A': [((1,1), 1)],
            'B': [((0,0), 1), ((0,1), -1)],
            'C': [((0,0), -1), ((1,0), 1)] # C11 and C21
        },
        # S5
        {
            'A': [((0,0), 1), ((0,1), 1)],
            'B': [((1,1), 1)],
            'C': [((0,1), 1), ((1,1), -1)] # C12 and C22
        },
        # S6
        {
            'A': [((1,0), 1), ((0,0), -1)],
            'B': [((0,0), 1), ((0,1), 1)],
            'C': [((1,0), 1)] # C21
        },
        # S7
        {
            'A': [((0,1), 1), ((1,1), -1)],
            'B': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((0,1), 1)] # C11 and C12
        }
    ]
    
    # Now, iterate over the 7 block terms (s) and the 7 scalar terms (t) within each.
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # Get the 2x2 Strassen coefficients for term t
            # u_t is a vector of 4 coeffs for A
            # v_t is a vector of 4 coeffs for B
            # w_t is a vector of 4 coeffs for C
            
            # For each A block involved in S_def[s]['A']:
            for (bi, bj), coeff_a in S_def[s]['A']:
                # Get global indices for this A block
                a_indices = get_block_indices(bi, bj, 'A')
                # For each local index l in 0..3:
                for l in range(4):
                    if U2[l, t] != 0:
                        U4[a_indices[l], col] += coeff_a * U2[l, t]
            
            # For each B block involved in S_def[s]['B']:
            for (bi, bj), coeff_b in S_def[s]['B']:
                # Get global indices for this B block
                # Note: B block indices (bi, bj) correspond to rows bi, cols bj.
                b_indices = get_block_indices(bi, bj, 'B')
                for l in range(4):
                    if V2[l, t] != 0:
                        V4[b_indices[l], col] += coeff_b * V2[l, t]
            
            # For each C block involved in S_def[s]['C']:
            for (bi, bk), coeff_c in S_def[s]['C']:
                # Get global indices for this C block
                # Note: C block indices (bi, bk) correspond to rows bi, cols bk.
                c_indices = get_block_indices(bi, bk, 'C')
                for l in range(4):
                    if W2[l, t] != 0:
                        W4[c_indices[l], col] += coeff_c * W2[l, t]
    
    # Convert to lists as required by interface
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

# EVOLVE-BLOCK-END
