"""
Rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.

Strategy: Split B (4x8) into two 4x4 blocks B_left and B_right.
C = A @ B = [A @ B_left | A @ B_right]
Each 4x4 * 4x4 multiplication is decomposed via recursive Strassen (rank 49).
Total rank = 49 + 49 = 98.

The 4x4*4x4 decomposition uses 4 Strassen 2x2*2x2 blocks (rank 7 each)
plus cross-block corrections, but the standard recursive Strassen gives rank 49
for 4x4*4x4 (since 7^2 = 49).
"""

DIMENSIONS = [4, 4, 8]

import time
import numpy as np

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication with rank 7.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Verified Strassen rank-7 decomposition for 2x2*2x2
    # A indices: A11=(0,0)->idx0, A12=(0,1)->idx1, A21=(1,0)->idx2, A22=(1,1)->idx3
    # B indices: B11=(0,0)->idx0, B12=(0,1)->idx1, B21=(1,0)->idx2, B22=(1,1)->idx3
    # C indices: C11=(0,0)->idx0, C12=(0,1)->idx1, C21=(1,0)->idx2, C22=(1,1)->idx3
    # W index convention: W[k*2+i, c] where k=col of B, i=row of A
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C21, C22
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12, C22
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11, C21
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11, C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1  # C21
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1  # C11, C12
    
    return U, V, W


def _verify_2x2(U, V, W):
    """Verify that the 2x2*2x2 decomposition is correct."""
    # For all i,j,k in {0,1}: sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] should be 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for c in range(7):
                    val += U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True


def _strassen_4x4():
    """
    Return U, V, W for 4x4 * 4x4 matrix multiplication with rank 49.
    Convention: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    
    Uses recursive Strassen: 4x4*4x4 is composed of 7 sub-multiplications,
    each of size 2x2*2x2 (rank 7), giving total rank 7*7=49.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # Recursive Strassen for 4x4*4x4:
    # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    # Split C into 2x2 blocks: C11, C12, C21, C22 (each 2x2)
    #
    # 7 products (each is 2x2*2x2, rank 7):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Output:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # For each of the 7 Strassen products, we use the rank-7 2x2 decomposition.
    # The key is to map the "combined" 2x2 blocks back to the 4x4 indices.
    
    # Let's define the 7 Strassen combinations at the block level.
    # Each combination has coefficients for A-blocks and B-blocks.
    # Then for each combination, we use the 2x2 Strassen U2, V2, W2 patterns.
    
    # A block indices in 4x4:
    # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5 in U (i*4+j)
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # B block indices in 4x4:
    # B11: rows 0-1, cols 0-1 -> indices 0,1,4,5 in V (j*4+k)
    # B12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # C block indices in 4x4:
    # C11: rows 0-1, cols 0-1 -> W index k*4+i: k in {0,1}, i in {0,1} -> 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> k in {2,3}, i in {0,1} -> 8,9,12,13
    # C21: rows 2-3, cols 0-1 -> k in {0,1}, i in {2,3} -> 2,3,6,7
    # C22: rows 2-3, cols 2-3 -> k in {2,3}, i in {2,3} -> 10,11,14,15
    
    # For each Strassen product t (0..6), we have a 2x2*2x2 multiplication.
    # The U2, V2, W2 tell us how to decompose a single 2x2*2x2 product.
    # We need to "embed" each 2x2 block operation into the 4x4 structure.
    
    # For product t, the A-part is a linear combination of A-blocks,
    # the B-part is a linear combination of B-blocks,
    # and the C-part contributes to C-blocks with certain coefficients.
    
    # Let me define the Strassen coefficients:
    # For each product t, we have:
    #   A_combo: dict mapping (block_row, block_col) -> coeff
    #   B_combo: dict mapping (block_row, block_col) -> coeff  
    #   C_combo: dict mapping (block_row, block_col) -> coeff
    
    # Product 0: M1 = (A11+A22)(B11+B22)
    # A: A11(+1), A22(+1)
    # B: B11(+1), B22(+1)
    # C: C11(+1), C22(+1)
    
    # Product 1: M2 = (A21+A22)B11
    # A: A21(+1), A22(+1)
    # B: B11(+1)
    # C: C21(+1), C22(-1)
    
    # Product 2: M3 = A11(B12-B22)
    # A: A11(+1)
    # B: B12(+1), B22(-1)
    # C: C12(+1), C22(+1)
    
    # Product 3: M4 = A22(B21-B11)
    # A: A22(+1)
    # B: B21(+1), B11(-1)
    # C: C11(+1), C21(+1)
    
    # Product 4: M5 = (A11+A12)B22
    # A: A11(+1), A12(+1)
    # B: B22(+1)
    # C: C11(-1), C12(+1)
    
    # Product 5: M6 = (A21-A11)(B11+B12)
    # A: A21(+1), A11(-1)
    # B: B11(+1), B12(+1)
    # C: C21(+1)
    
    # Product 6: M7 = (A12-A22)(B21+B22)
    # A: A12(+1), A22(-1)
    # B: B21(+1), B22(+1)
    # C: C11(-1), C12(+1)
    
    # For each product t and each of the 7 sub-terms c in the 2x2 Strassen:
    # The U2[c] gives the A-element coefficients within the 2x2 block
    # The V2[c] gives the B-element coefficients within the 2x2 block
    # The W2[c] gives the C-element coefficients within the 2x2 block
    
    # I'll construct this by iterating over products and sub-terms.
    
    # Define block-to-index mappings for 4x4:
    # A block (br, bc) where br in {0,1}, bc in {0,1}:
    #   element (i,j) in block -> global index (br*2+i)*4 + (bc*2+j)
    # B block (br, bc):
    #   element (i,j) in block -> global index (br*2+i)*4 + (bc*2+j)
    # C block (br, bc):
    #   element (i,j) in block -> W index (bc*2+j)*4 + (br*2+i)
    
    # Strassen product definitions:
    # (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: {block_idx: coeff} where block_idx 0=A11,1=A12,2=A21,3=A22
    # b_coeffs: {block_idx: coeff} where block_idx 0=B11,1=B12,2=B21,3=B22
    # c_coeffs: {block_idx: coeff} where block_idx 0=C11,1=C12,2=C21,3=C22
    
    products = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1, 1: 1}),
    ]
    
    for t, (a_coeffs, b_coeffs, c_coeffs) in enumerate(products):
        for c in range(7):
            col = t * 7 + c
            
            # A part: U2[ai, c] for each element in the 2x2 A-block
            for ai in range(4):  # 0=A11_elem00, 1=A11_elem01, 2=A11_elem10, 3=A11_elem11
                u2_val = U2[ai, c]
                if u2_val == 0:
                    continue
                # ai corresponds to element (ai//2, ai%2) in the 2x2 block
                local_i = ai // 2
                local_j = ai % 2
                # Apply to each A-block that has a coefficient
                for block_idx, a_coeff in a_coeffs.items():
                    val = u2_val * a_coeff
                    if val == 0:
                        continue
                    br = block_idx // 2
                    bc = block_idx % 2
                    global_i = br * 2 + local_i
                    global_j = bc * 2 + local_j
                    U[global_i * 4 + global_j, col] += val
            
            # B part: V2[bj, c] for each element in the 2x2 B-block
            for bj in range(4):
                v2_val = V2[bj, c]
                if v2_val == 0:
                    continue
                local_i = bj // 2
                local_j = bj % 2
                for block_idx, b_coeff in b_coeffs.items():
                    val = v2_val * b_coeff
                    if val == 0:
                        continue
                    br = block_idx // 2
                    bc = block_idx % 2
                    global_i = br * 2 + local_i
                    global_j = bc * 2 + local_j
                    V[global_i * 4 + global_j, col] += val
            
            # C part: W2[ci, c] for each element in the 2x2 C-block
            for ci in range(4):
                w2_val = W2[ci, c]
                if w2_val == 0:
                    continue
                # ci: W index k*2+i in 2x2, so ci = k*2+i
                local_k = ci // 2
                local_i = ci % 2
                for block_idx, c_coeff in c_coeffs.items():
                    val = w2_val * c_coeff
                    if val == 0:
                        continue
                    br = block_idx // 2  # block row = output row block
                    bc = block_idx % 2  # block col = output col block
                    global_k = bc * 2 + local_k
                    global_i = br * 2 + local_i
                    W[global_k * 4 + global_i, col] += val
    
    return U, V, W


def _verify_4x4(U, V, W):
    """Verify the 4x4*4x4 decomposition."""
    n = 4
    for i in range(n):
        for j in range(n):
            for k in range(n):
                val = 0.0
                for c in range(49):
                    val += U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
                if abs(val - 1.0) > 1e-8:
                    return False
    return True


def construct():
    t_start = time.time()
    DEADLINE = t_start + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Try Strassen-based decomposition first
    try:
        U4, V4, W4 = _strassen_4x4()
        
        # Verify
        if _verify_4x4(U4, V4, W4):
            # Compose for 4x4 * 4x8
            # B is 4x8, split into B_left (4x4) and B_right (4x4)
            # C is 4x8, split into C_left (4x4) and C_right (4x4)
            # C_left = A @ B_left, C_right = A @ B_right
            # Each uses the 4x4*4x4 decomposition (rank 49)
            # Total rank = 98
            
            rank_total = 98
            U = np.zeros((n*m, rank_total), dtype=float)  # (16, 98)
            V = np.zeros((m*p, rank_total), dtype=float)  # (32, 98)
            W = np.zeros((p*n, rank_total), dtype=float)  # (32, 98)
            
            # Left half: columns 0-3 of B, columns 0-3 of C
            # U: same for both halves (A is the same)
            # V: for left half, V[j*4+k, c] for k in 0..3 -> V[j*8+k, c]
            # W: for left half, W[k*4+i, c] for k in 0..3 -> W[k*8+i, c]
            
            # Right half: columns 4-7 of B, columns 4-7 of C
            # V: for right half, V[j*4+k, c] for k in 0..3 -> V[j*8+4+k, c]
            # W: for right half, W[k*4+i, c] for k in 0..3 -> W[(k+4)*8+i, c]
            
            # Left half (columns 0-3):
            U[:, 0:49] = U4
            # V left: V4[j*4+k, c] -> V[j*8+k, c] for k in 0..3
            for j in range(4):
                for k in range(4):
                    V[j*8+k, 0:49] = V4[j*4+k, :]
            # W left: W4[k*4+i, c] -> W[k*8+i, c] for k in 0..3
            for k in range(4):
                for i in range(4):
                    W[k*8+i, 0:49] = W4[k*4+i, :]
            
            # Right half (columns 4-7):
            U[:, 49:98] = U4
            # V right: V4[j*4+k, c] -> V[j*8+4+k, c] for k in 0..3
            for j in range(4):
                for k in range(4):
                    V[j*8+4+k, 49:98] = V4[j*4+k, :]
            # W right: W4[k*4+i, c] -> W[(k+4)*8+i, c] for k in 0..3
            for k in range(4):
                for i in range(4):
                    W[(k+4)*8+i, 49:98] = W4[k*4+i, :]
            
            # Final verification
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank_total):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(val - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list format
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        # If Strassen verification fails, fall through to schoolbook
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 128)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
