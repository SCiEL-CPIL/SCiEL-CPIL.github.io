import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _get_indices(n, block):
    """
    Get the flat indices for a n/2 x n/2 block in an n x n matrix.
    block: '00', '01', '10', '11'
    """
    h = n // 2
    r0, c0 = 0, 0
    if block == '00':
        r0, c0 = 0, 0
    elif block == '01':
        r0, c0 = 0, h
    elif block == '10':
        r0, c0 = h, 0
    elif block == '11':
        r0, c0 = h, h
    
    indices = []
    for i in range(h):
        for j in range(h):
            indices.append((r0 + i) * n + (c0 + j))
    return indices

def _strassen_decomp(n):
    """
    Returns U, V, W matrices for n x n x n matrix multiplication tensor.
    U: (n*n, rank)
    V: (n*n, rank)
    W: (n*n, rank)
    """
    if n == 1:
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W
    
    h = n // 2
    U_sub, V_sub, W_sub = _strassen_decomp(h)
    sub_rank = U_sub.shape[1]
    total_rank = 7 * sub_rank
    
    U = np.zeros((n*n, total_rank))
    V = np.zeros((n*n, total_rank))
    W = np.zeros((n*n, total_rank))
    
    # Precompute indices for blocks
    A_00 = _get_indices(n, '00')
    A_01 = _get_indices(n, '01')
    A_10 = _get_indices(n, '10')
    A_11 = _get_indices(n, '11')
    
    B_00 = A_00 # Same structure
    B_01 = A_01
    B_10 = A_10
    B_11 = A_11
    
    C_00 = A_00
    C_01 = A_01
    C_10 = A_10
    C_11 = A_11
    
    # Strassen steps
    # M_k = (sum A_blocks * coeff) * (sum B_blocks * coeff)
    # Contrib to C_blocks with coeffs
    
    steps = [
        # (A_sources, B_sources, C_destinations)
        # A_sources: list of (indices, coeff)
        # M1 = (A00 + A11) (B00 + B11)
        ([(A_00, 1), (A_11, 1)], [(B_00, 1), (B_11, 1)], [(C_00, 1), (C_11, 1)]),
        # M2 = (A10 + A11) B00
        ([(A_10, 1), (A_11, 1)], [(B_00, 1)], [(C_10, 1)]),
        # M3 = A00 (B01 - B11)
        ([(A_00, 1)], [(B_01, 1), (B_11, -1)], [(C_01, 1)]),
        # M4 = A11 (B10 - B00)
        ([(A_11, 1)], [(B_10, 1), (B_00, -1)], [(C_10, 1), (C_00, -1)]),
        # M5 = (A00 + A01) B11
        ([(A_00, 1), (A_01, 1)], [(B_11, 1)], [(C_01, 1), (C_00, -1)]),
        # M6 = (A10 - A00) (B00 + B01)
        ([(A_10, 1), (A_00, -1)], [(B_00, 1), (B_01, 1)], [(C_11, 1)]),
        # M7 = (A01 - A11) (B10 + B11)
        ([(A_01, 1), (A_11, -1)], [(B_10, 1), (B_11, 1)], [(C_00, 1), (C_11, -1)]),
    ]
    
    # Correction for C coefficients based on standard Strassen:
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # My steps above defined contributions to C from each M_k.
    # Let's re-verify the C_destinations for each step.
    
    # M1 contributes to C00 (+1) and C11 (+1)
    # M2 contributes to C10 (+1)
    # M3 contributes to C01 (+1) and C11 (-1)
    # M4 contributes to C10 (+1) and C00 (-1)
    # M5 contributes to C01 (+1) and C00 (-1)
    # M6 contributes to C11 (+1)
    # M7 contributes to C00 (+1) and C11 (-1)
    
    # Let's update the steps list with correct C contributions.
    steps = [
        # M1
        ([(A_00, 1), (A_11, 1)], [(B_00, 1), (B_11, 1)], [(C_00, 1), (C_11, 1)]),
        # M2
        ([(A_10, 1), (A_11, 1)], [(B_00, 1)], [(C_10, 1)]),
        # M3
        ([(A_00, 1)], [(B_01, 1), (B_11, -1)], [(C_01, 1), (C_11, -1)]),
        # M4
        ([(A_11, 1)], [(B_10, 1), (B_00, -1)], [(C_10, 1), (C_00, -1)]),
        # M5
        ([(A_00, 1), (A_01, 1)], [(B_11, 1)], [(C_01, 1), (C_00, -1)]),
        # M6
        ([(A_10, 1), (A_00, -1)], [(B_00, 1), (B_01, 1)], [(C_11, 1)]),
        # M7
        ([(A_01, 1), (A_11, -1)], [(B_10, 1), (B_11, 1)], [(C_00, 1), (C_11, -1)]),
    ]
    
    for k, (a_src, b_src, c_dst) in enumerate(steps):
        for r in range(sub_rank):
            col = k * sub_rank + r
            
            # U
            for idxs, coeff in a_src:
                U[idxs, col] += coeff * U_sub[:, r]
            
            # V
            for idxs, coeff in b_src:
                V[idxs, col] += coeff * V_sub[:, r]
            
            # W
            for idxs, coeff in c_dst:
                W[idxs, col] += coeff * W_sub[:, r]
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4 x 4x4 (Rank 49)
    U_444, V_444, W_444 = _strassen_decomp(4)
    
    # Part 2: 4x4 x 4x1 (Rank 4)
    # U: (16, 4), V: (20, 4), W: (20, 4)
    U_441 = np.zeros((16, 4))
    V_441 = np.zeros((20, 4))
    W_441 = np.zeros((20, 4))
    
    for j in range(4):
        # u_j: 1 at A_{ij} for all i
        for i in range(4):
            idx_a = i * 4 + j
            U_441[idx_a, j] = 1.0
            
        # v_j: 1 at B_{j,4}
        idx_b = j * 5 + 4
        V_441[idx_b, j] = 1.0
        
        # w_j: 1 at C_{i,4} for all i
        for i in range(4):
            idx_c = 4 * 4 + i # k*n+i = 4*4+i
            W_441[idx_c, j] = 1.0
            
    # Combine
    # U: (16, 49) + (16, 4) -> (16, 53)
    U = np.hstack([U_444, U_441])
    
    # V: (16, 49) + (20, 4) -> (20, 53)
    # Pad V_444 to (20, 49)
    V_444_padded = np.zeros((20, 49))
    V_444_padded[:16, :] = V_444
    V = np.hstack([V_444_padded, V_441])
    
    # W: (16, 49) + (20, 4) -> (20, 53)
    # Pad W_444 to (20, 49)
    W_444_padded = np.zeros((20, 49))
    W_444_padded[:16, :] = W_444
    W = np.hstack([W_444_padded, W_441])
    
    # Convert to list of lists as required by interface?
    # "return {'U', 'V', 'W'}: factor matrices ... with integer/half-integer entries"
    # The interface in seed uses lists. I will return numpy arrays or lists.
    # Seed returns lists. I'll stick to lists to be safe, or numpy if allowed.
    # "Python standard library, NumPy and SciPy are available."
    # Usually numpy arrays are accepted if the validator can handle them.
    # But seed uses lists. I will convert to lists of floats/ints.
    
    def to_list(arr):
        return arr.tolist()
        
    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
