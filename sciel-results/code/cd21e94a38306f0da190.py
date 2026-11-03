"""
Strategy: Recursive Strassen for 4x4x4 blocks, then extend to 4x4x8.
The 4x4x8 matrix multiplication can be decomposed into two 4x4x4 multiplications
(left half and right half of the B matrix). Using Strassen recursion on each 4x4x4
gives rank 49 per block, totaling rank 98.

This is a clean, verifiable construction that should beat the schoolbook rank of 128.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _get_target_tensor():
    """Construct the target matrix multiplication tensor T[a,b,c]."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _build_strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication with rank 49.
    Uses recursive Strassen: 7 multiplications of 2x2, each with rank 7.
    """
    n = 4
    dim = n * n
    
    # Define the 2x2 Strassen terms (7 terms, each with u, v, w vectors of length 4)
    # Indexing: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    terms_2x2 = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 0]),
        # M2 = (A10+A11)B00 -> C10 += M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0]),
        # M3 = A00(B01-B11) -> C11 -= M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 0, 0]),
        # M4 = A11(B10-B00) -> C00 -= M4, C10 += M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 0]),
        # M5 = (A00+A01)B11 -> C01 += M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]),
        # M6 = (A10-A00)(B00+B01) -> C00 += M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]),
        # M7 = (A01-A11)(B10+B11) -> C01 += M7, C11 += M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 0]),
    ]
    
    # Now we need to apply Strassen at the top level (4x4 as 2x2 blocks of 2x2 matrices)
    # Top-level Strassen: 7 multiplications of 2x2 matrices
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00, B01, B10, B11 (similar)
    # C blocks: C00, C01, C10, C11
    
    # Top-level Strassen coefficients
    # M1_top = (A00+A11)(B00+B11)
    # M2_top = (A10+A11)B00
    # M3_top = A00(B01-B11)
    # M4_top = A11(B10-B00)
    # M5_top = (A00+A01)B11
    # M6_top = (A10-A00)(B00+B01)
    # M7_top = (A01-A11)(B10+B11)
    
    # For each top-level term, we need to combine it with bottom-level Strassen terms
    # Total rank = 7 * 7 = 49
    
    U = np.zeros((dim, 49))
    V = np.zeros((dim, 49))
    W = np.zeros((dim, 49))
    
    # Block index to global index mapping
    # Block (br, bc) contains elements at global indices:
    # (br*2 + lr) * 4 + (bc*2 + lc) for lr, lc in 0..1
    def block_to_indices(br, bc):
        """Return list of 4 global indices for block (br, bc)."""
        indices = []
        for lr in range(2):
            for lc in range(2):
                gr = br * 2 + lr
                gc = bc * 2 + lc
                indices.append(gr * 4 + gc)
        return indices
    
    # Top-level term definitions
    # Each term has coefficients for A blocks, B blocks, and which C blocks it contributes to
    top_terms = [
        # (u_blocks, v_blocks, w_contributions)
        # u_blocks: [coeff for A00, A01, A10, A11]
        # v_blocks: [coeff for B00, B01, B10, B11]
        # w_contributions: dict mapping C_block_idx -> coeff
        # M1_top
        {
            'u_blocks': [1, 0, 0, 1],
            'v_blocks': [1, 0, 0, 1],
            'w_contributions': {0: 1, 3: 1}  # C00 += M1, C11 += M1
        },
        # M2_top
        {
            'u_blocks': [0, 0, 1, 1],
            'v_blocks': [1, 0, 0, 0],
            'w_contributions': {2: 1}  # C10 += M2
        },
        # M3_top
        {
            'u_blocks': [1, 0, 0, 0],
            'v_blocks': [0, 1, 0, -1],
            'w_contributions': {3: -1}  # C11 -= M3
        },
        # M4_top
        {
            'u_blocks': [0, 0, 0, 1],
            'v_blocks': [0, 0, 1, -1],
            'w_contributions': {0: -1, 2: 1}  # C00 -= M4, C10 += M4
        },
        # M5_top
        {
            'u_blocks': [1, 1, 0, 0],
            'v_blocks': [0, 0, 0, 1],
            'w_contributions': {1: 1}  # C01 += M5
        },
        # M6_top
        {
            'u_blocks': [-1, 0, 1, 0],
            'v_blocks': [1, 1, 0, 0],
            'w_contributions': {0: 1}  # C00 += M6
        },
        # M7_top
        {
            'u_blocks': [0, 1, 0, -1],
            'v_blocks': [0, 0, 1, 1],
            'w_contributions': {1: 1, 3: 1}  # C01 += M7, C11 += M7
        },
    ]
    
    col = 0
    for top_term in top_terms:
        u_top = top_term['u_blocks']
        v_top = top_term['v_blocks']
        w_top = top_term['w_contributions']
        
        for bot_term in terms_2x2:
            u_bot, v_bot, w_bot = bot_term
            
            u_full = np.zeros(dim)
            v_full = np.zeros(dim)
            w_full = np.zeros(dim)
            
            # Expand u_full
            for b_idx in range(4):
                if u_top[b_idx] != 0:
                    br = b_idx // 2
                    bc = b_idx % 2
                    indices = block_to_indices(br, bc)
                    for l in range(4):
                        if u_bot[l] != 0:
                            u_full[indices[l]] += u_top[b_idx] * u_bot[l]
            
            # Expand v_full
            for b_idx in range(4):
                if v_top[b_idx] != 0:
                    br = b_idx // 2
                    bc = b_idx % 2
                    indices = block_to_indices(br, bc)
                    for l in range(4):
                        if v_bot[l] != 0:
                            v_full[indices[l]] += v_top[b_idx] * v_bot[l]
            
            # Expand w_full
            for c_block_idx, coeff in w_top.items():
                if coeff != 0:
                    br = c_block_idx // 2
                    bc = c_block_idx % 2
                    indices = block_to_indices(br, bc)
                    for l in range(4):
                        if w_bot[l] != 0:
                            w_full[indices[l]] += coeff * w_bot[l]
            
            U[:, col] = u_full
            V[:, col] = v_full
            W[:, col] = w_full
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_k U[:,k] V[:,k] W[:,k] == T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _build_4x4x8_from_4x4x4():
    """
    Build 4x4x8 decomposition by combining two 4x4x4 decompositions.
    Left half: k=0..3, Right half: k=4..7
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    U4, V4, W4 = _build_strassen_4x4x4()
    rank4 = U4.shape[1]  # 49
    
    total_rank = 2 * rank4  # 98
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # Left half (k=0..3)
    U_full[:, 0:rank4] = U4
    
    for r in range(rank4):
        for idx in range(16):
            val = V4[idx, r]
            if val != 0:
                j = idx // 4
                k_local = idx % 4
                v_idx = j * 8 + k_local
                V_full[v_idx, r] += val
                
            val = W4[idx, r]
            if val != 0:
                k_local = idx // 4
                i = idx % 4
                w_idx = k_local * 4 + i
                W_full[w_idx, r] += val
    
    # Right half (k=4..7)
    U_full[:, rank4:2*rank4] = U4
    
    for r in range(rank4):
        for idx in range(16):
            val = V4[idx, r]
            if val != 0:
                j = idx // 4
                k_local = idx % 4
                v_idx = j * 8 + (k_local + 4)
                V_full[v_idx, r + rank4] += val
                
            val = W4[idx, r]
            if val != 0:
                k_local = idx % 4
                i = idx // 4
                w_idx = (k_local + 4) * 4 + i
                W_full[w_idx, r + rank4] += val
    
    return U_full, V_full, W_full

def _to_half_integer_list(mat):
    """Convert numpy array to list of half-integers (or integers)."""
    mat2 = mat * 2
    rounded = np.round(mat2)
    if np.allclose(mat2, rounded, atol=1e-6):
        return (rounded / 2).tolist()
    rounded_int = np.round(mat)
    if np.allclose(mat, rounded_int, atol=1e-6):
        return rounded_int.astype(int).tolist()
    return mat.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build the 4x4x8 decomposition using recursive Strassen
    U, V, W = _build_4x4x8_from_4x4x4()
    
    # Verify
    T = _get_target_tensor()
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank_school = n * m * p
        U_s = np.zeros((n*m, rank_school))
        V_s = np.zeros((m*p, rank_school))
        W_s = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1.0
                    V_s[j*p + k, col] = 1.0
                    W_s[k*n + i, col] = 1.0
                    col += 1
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }
    
    # Check deadline
    if time.time() > deadline:
        return {
            "U": _to_half_integer_list(U),
            "V": _to_half_integer_list(V),
            "W": _to_half_integer_list(W)
        }
    
    # Convert to half-integers
    U_out = _to_half_integer_list(U)
    V_out = _to_half_integer_list(V)
    W_out = _to_half_integer_list(W)
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
