"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2x2():
    """
    Returns U, V, W for rank-7 decomposition of 2x2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: i*2+j for A, i*2+j for B, i*2+j for C.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00 + A11)(B00 + B11)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # P2 = (A10 + A11) B00
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, 0]
    
    # P3 = A00 (B01 - B11)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, 0]
    
    # P4 = A11 (B10 - B00)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 0, 1, -1]
    W[:, 3] = [1, 0, 0, 0]
    
    # P5 = (A00 + A01) B11
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 0, 0]
    
    # P6 = (A10 - A00)(B00 + B01)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # P7 = (A01 - A11)(B10 + B11)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Builds rank-49 decomposition for 4x4x4 matrix multiplication.
    U shape (16, 49), V shape (16, 49), W shape (16, 49).
    """
    U_base, V_base, W_base = _strassen_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Block index mappings for 4x4 matrix (row-major)
    # Block (i,j) of 2x2 blocks of 2x2 matrices
    # Block 00: indices 0,1,4,5
    # Block 01: indices 2,3,6,7
    # Block 10: indices 8,9,12,13
    # Block 11: indices 10,11,14,15
    
    # Map 2x2 index (0,1,2,3) to 4x4 index for each block
    def block_indices(i, j):
        # i, j in {0, 1}
        # Base index of top-left of 2x2 block
        # Block (0,0) starts at 0
        # Block (0,1) starts at 2
        # Block (1,0) starts at 8
        # Block (1,1) starts at 10
        base = i * 8 + j * 2
        # Indices: base, base+1, base+4, base+5
        return [base, base + 1, base + 4, base + 5]
    
    # Strassen coefficients for A, B, C blocks
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    # A: 00:1, 11:1
    # B: 00:1, 11:1
    # C: 00:1, 11:1
    # Term 1: M2 = (A10 + A11) B00
    # A: 10:1, 11:1
    # B: 00:1
    # C: 10:1
    # Term 2: M3 = A00 (B01 - B11)
    # A: 00:1
    # B: 01:1, 11:-1
    # C: 01:1
    # Term 3: M4 = A11 (B10 - B00)
    # A: 11:1
    # B: 10:1, 00:-1
    # C: 00:1
    # Term 4: M5 = (A00 + A01) B11
    # A: 00:1, 01:1
    # B: 11:1
    # C: 00:-1
    # Term 5: M6 = (A10 - A00)(B00 + B01)
    # A: 10:1, 00:-1
    # B: 00:1, 01:1
    # C: 11:1
    # Term 6: M7 = (A01 - A11)(B10 + B11)
    # A: 01:1, 11:-1
    # B: 10:1, 11:1
    # C: 00:1
    
    # Note: The W coefficients in _strassen_2x2x2 are for the 2x2 C matrix.
    # The block-level C contributions are determined by how M_k maps to C blocks.
    # M1 -> C00 (+1), C11 (+1)
    # M2 -> C10 (+1)
    # M3 -> C01 (+1)
    # M4 -> C00 (+1)
    # M5 -> C00 (-1)
    # M6 -> C11 (+1)
    # M7 -> C00 (+1)
    
    # We need to combine the block-level C coefficients with the base W coefficients.
    # The base W coefficients distribute the result within the 2x2 block.
    
    strassen_terms = [
        # (A_blocks: dict, B_blocks: dict, C_blocks: dict)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    for k in range(7):
        A_blocks, B_blocks, C_blocks = strassen_terms[k]
        
        for r in range(7):
            col = k * 7 + r
            
            # Expand U
            for (bi, bj), coeff in A_blocks.items():
                idxs = block_indices(bi, bj)
                for local_idx, global_idx in enumerate(idxs):
                    U[global_idx, col] += coeff * U_base[local_idx, r]
            
            # Expand V
            for (bi, bj), coeff in B_blocks.items():
                idxs = block_indices(bi, bj)
                for local_idx, global_idx in enumerate(idxs):
                    V[global_idx, col] += coeff * V_base[local_idx, r]
            
            # Expand W
            for (bi, bj), coeff in C_blocks.items():
                idxs = block_indices(bi, bj)
                for local_idx, global_idx in enumerate(idxs):
                    W[global_idx, col] += coeff * W_base[local_idx, r]
                    
    return U, V, W

def _build_4x4x2_schoolbook():
    """
    Builds rank-32 decomposition for 4x4x2 matrix multiplication.
    U shape (16, 32), V shape (24, 32), W shape (24, 32).
    Note: V is for B (4x6), so size 24. W is for C (4x6), so size 24.
    This function returns the parts for the last 2 columns.
    """
    U = np.zeros((16, 32))
    V = np.zeros((24, 32))
    W = np.zeros((24, 32))
    
    col = 0
    # B columns are 4 and 5
    # C columns are 4 and 5
    for c_col in [4, 5]:
        for i in range(4):
            for k in range(4):
                U[i * 4 + k, col] = 1
                V[k * 6 + c_col, col] = 1
                W[i * 4 + c_col, col] = 1
                col += 1
                
    return U, V, W

def construct():
    U1, V1, W1 = _build_4x4x4_strassen()
    U2, V2, W2 = _build_4x4x2_schoolbook()
    
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

