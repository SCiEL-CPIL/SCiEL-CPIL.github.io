import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _verify(U, V, W, T, tol=1e-9):
    """Verify tensor decomposition."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    r = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _build_strassen_4x4x4():
    """
    Build a rank-49 decomposition for 4x4x4 matrix multiplication using 
    block-Strassen (4 blocks of 2x2, each using 7 Strassen multiplications).
    """
    n = 4
    rank = 49
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)
    
    # Strassen 2x2x2 terms (7 terms)
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    s_terms = [
        # M1
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        # M3
        ([(0,1)], [(1,1),(3,-1)], [(1,1)]),
        # M4
        ([(3,1)], [(2,1),(0,-1)], [(3,1)]),
        # M5
        ([(0,1),(1,1)], [(3,1)], [(1,1)]),
        # M6
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(0,1)]),
        # M7
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)])
    ]
    
    # Block mapping for 4x4
    # Blocks: 00, 01, 10, 11
    # Row block r, Col block c
    # Linear index for element (i, j) in block (rb, cb) is (rb*2+i)*4 + (cb*2+j)
    
    # For each of the 7 block-level Strassen terms
    for k in range(7):
        u_entries_block, v_entries_block, w_entries_block = s_terms[k]
        
        # We need to expand each block-level entry into 4 element-level entries
        # using the 7 element-level Strassen terms.
        
        # u_entries_block contains (block_idx, sign)
        # block_idx 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        # For each element-level Strassen term s in 0..6
        for s in range(7):
            t = k * 7 + s
            u_entries_elem, v_entries_elem, w_entries_elem = s_terms[s]
            
            # Accumulate U
            for b_idx, b_sign in u_entries_block:
                rb, cb = b_idx // 2, b_idx % 2
                for e_idx, e_sign in u_entries_elem:
                    er, ec = e_idx // 2, e_idx % 2
                    i = rb * 2 + er
                    j = cb * 2 + ec
                    idx = i * 4 + j
                    U[idx, t] += b_sign * e_sign
            
            # Accumulate V
            for b_idx, b_sign in v_entries_block:
                rb, cb = b_idx // 2, b_idx % 2
                for e_idx, e_sign in v_entries_elem:
                    er, ec = e_idx // 2, e_idx % 2
                    i = rb * 2 + er
                    j = cb * 2 + ec
                    idx = i * 4 + j
                    V[idx, t] += b_sign * e_sign
            
            # Accumulate W
            for b_idx, b_sign in w_entries_block:
                rb, cb = b_idx // 2, b_idx % 2
                for e_idx, e_sign in w_entries_elem:
                    er, ec = e_idx // 2, e_idx % 2
                    i = rb * 2 + er
                    j = cb * 2 + ec
                    idx = i * 4 + j
                    W[idx, t] += b_sign * e_sign

    # Verify 4x4x4
    T4 = np.zeros((16, 16, 16), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T4[i*4+j, j*4+k, k*4+i] = 1.0
    
    if not _verify(U, V, W, T4):
        return None, None, None
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Build the full 4x4x8 multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Schoolbook decomposition (rank 128) as fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook

    # Try to build a verified Strassen 4x4x4 block and extend to 4x4x8
    if time.time() < deadline - 5:
        U4, V4, W4 = _build_strassen_4x4x4()
        if U4 is not None:
            rank_block = U4.shape[1]
            # Extend to 4x4x8 by duplicating the block for the two 4-column output blocks
            U_full = np.zeros((n*m, 2*rank_block), dtype=np.float64)
            V_full = np.zeros((m*p, 2*rank_block), dtype=np.float64)
            W_full = np.zeros((p*n, 2*rank_block), dtype=np.float64)
            
            # Block 1: k=0..3 (B columns 0..3, C rows 0..3)
            U_full[:, 0:rank_block] = U4
            # V4 index b' = j*4 + k. V_full index b = j*8 + k.
            # So row b = 8j + k gets V4[4j + k].
            for j in range(4):
                for k in range(4):
                    V_full[8*j + k, 0:rank_block] = V4[4*j + k, :]
            
            # W4 index c' = k*4 + i. W_full index c = k*4 + i.
            # For k=0..3, c = 4k + i. This is the same index.
            W_full[0:16, 0:rank_block] = W4
            
            # Block 2: k=4..7 (B columns 4..7, C rows 4..7)
            U_full[:, rank_block:2*rank_block] = U4
            # V4 index b' = j*4 + k. V_full index b = j*8 + (4+k).
            # So row b = 8j + 4 + k gets V4[4j + k].
            for j in range(4):
                for k in range(4):
                    V_full[8*j + 4 + k, rank_block:2*rank_block] = V4[4*j + k, :]
            
            # W4 index c' = k*4 + i. W_full index c = (4+k)*4 + i = 16 + 4k + i.
            for k in range(4):
                for i in range(4):
                    W_full[16 + 4*k + i, rank_block:2*rank_block] = W4[4*k + i, :]
            
            if _verify(U_full, V_full, W_full, T):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 2*rank_block

    # Convert to half-integer lists
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
