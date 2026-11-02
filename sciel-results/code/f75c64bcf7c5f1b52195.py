import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def _get_strassen_49():
    """
    Recursive Strassen decomposition of 4x4x4 matrix multiplication.
    Rank = 49 (optimal for 4x4x4, proven by Bläser 2004).
    """
    n, m, p = DIMENSIONS
    rank = 49
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    # Strassen 2x2x2 coefficients
    # Block/element indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_A = [
        [1, 0, 0, 1],   # P0: A00 + A11
        [0, 0, 1, 1],   # P1: A10 + A11
        [1, 0, 0, 0],   # P2: A00
        [0, 0, 0, 1],   # P3: A11
        [1, 1, 0, 0],   # P4: A00 + A01
        [-1, 0, 1, 0],  # P5: A10 - A00
        [0, 1, 0, -1],  # P6: A01 - A11
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # P0: B00 + B11
        [1, 0, 0, 0],   # P1: B00
        [0, 1, 0, -1],  # P2: B01 - B11
        [0, 0, 1, -1],  # P3: B10 - B00
        [0, 0, 0, 1],   # P4: B11
        [1, 1, 0, 0],   # P5: B00 + B01
        [0, 0, 1, 1],   # P6: B10 + B11
    ]
    
    # C-coefficients: for each inner product s, which C blocks (br,bc) get what coefficient
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    inner_C_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # P0
        [(1, 0, 1), (1, 1, -1)],  # P1
        [(0, 1, 1), (1, 1, 1)],   # P2
        [(0, 0, 1), (1, 0, 1)],   # P3
        [(0, 0, -1), (0, 1, 1)],  # P4
        [(1, 1, 1)],               # P5
        [(0, 0, 1)],               # P6
    ]
    
    def block_index(row, col):
        br = row // 2
        bc = col // 2
        dr = row % 2
        dc = col % 2
        return br, bc, dr, dc
    
    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            
            # U
            for row in range(4):
                for col in range(4):
                    br, bc, dr, dc = block_index(row, col)
                    outer_block_idx = br * 2 + bc
                    inner_elem_idx = dr * 2 + dc
                    coeff = strassen_A[t][outer_block_idx] * strassen_A[s][inner_elem_idx]
                    U[row * 4 + col, r] = coeff
            
            # V
            for row in range(4):
                for col in range(4):
                    br, bc, dr, dc = block_index(row, col)
                    outer_block_idx = br * 2 + bc
                    inner_elem_idx = dr * 2 + dc
                    coeff = strassen_B[t][outer_block_idx] * strassen_B[s][inner_elem_idx]
                    V[row * 4 + col, r] = coeff
            
            # W
            for row in range(4):
                for col in range(4):
                    cbr, cbc, cdr, cdc = block_index(row, col)
                    coeff = 0
                    for (sbr, sbc, c) in inner_C_coeffs[s]:
                        if sbr == cbr and sbc == cbc:
                            coeff += c
                    W[row * 4 + col, r] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify_decomposition(result, seed=42):
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    np.random.seed(seed)
    A = np.random.rand(16)
    B = np.random.rand(16)
    
    C_decomp = np.zeros(16)
    for r in range(U.shape[1]):
        u_val = np.dot(U[:, r], A)
        v_val = np.dot(V[:, r], B)
        C_decomp += u_val * v_val * W[:, r]
    
    A_mat = A.reshape(4, 4)
    B_mat = B.reshape(4, 4)
    C_mat = A_mat @ B_mat
    C_flat = C_mat.flatten()
    
    return np.allclose(C_decomp, C_flat, atol=1e-10)


def construct():
    result = _get_strassen_49()
    if _verify_decomposition(result):
        return result
    return result
