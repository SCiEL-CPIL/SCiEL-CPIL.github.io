import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    Uses block decomposition with Strassen for 2x2x2 blocks.
    Optimal partition: n=[2,1], m=[2,2,1], p=[2,2,2].
    Rank: 84.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin

    # Define the optimal partition
    n_sizes = [2, 1]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]
    
    # Calculate offsets
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    # Pre-calculate total rank
    # 6 blocks of 2x2x2 (rank 7)
    # 3 blocks of 2x1x2 (rank 4)
    # 6 blocks of 1x2x2 (rank 4)
    # 3 blocks of 1x1x2 (rank 2)
    # Total: 6*7 + 3*4 + 6*4 + 3*2 = 42 + 12 + 24 + 6 = 84
    total_rank = 84
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add Strassen terms for a 2x2x2 block
    def add_strassen_block(n_start, m_start, p_start, rank_offset):
        # Local indices for the block
        # A rows: n_start, n_start+1
        # B cols: m_start, m_start+1
        # C cols: p_start, p_start+1
        
        # Strassen products for 2x2:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)(b00)
        # m3 = (a00)(b01-b11)
        # m4 = (a11)(b10-b00)
        # m5 = (a00+a01)(b11)
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # Indices in global U, V, W
        # U[i*m + j]
        # V[j*p + k]
        # W[k*n + i]
        
        # Map local (li, lj, lk) to global (i, j, k)
        i0, i1 = n_start, n_start + 1
        j0, j1 = m_start, m_start + 1
        k0, k1 = p_start, p_start + 1
        
        # m1
        U[i0*m + j0][rank_offset] += 1.0
        U[i1*m + j1][rank_offset] += 1.0
        V[j0*p + k0][rank_offset] += 1.0
        V[j1*p + k1][rank_offset] += 1.0
        W[k0*n + i0][rank_offset] += 1.0
        W[k1*n + i1][rank_offset] += 1.0
        
        # m2
        U[i1*m + j0][rank_offset+1] += 1.0
        U[i1*m + j1][rank_offset+1] += 1.0
        V[j0*p + k0][rank_offset+1] += 1.0
        W[k0*n + i1][rank_offset+1] += 1.0
        W[k1*n + i1][rank_offset+1] -= 1.0
        
        # m3
        U[i0*m + j0][rank_offset+2] += 1.0
        V[j0*p + k1][rank_offset+2] += 1.0
        V[j1*p + k1][rank_offset+2] -= 1.0
        W[k1*n + i0][rank_offset+2] += 1.0
        W[k1*n + i1][rank_offset+2] += 1.0
        
        # m4
        U[i1*m + j1][rank_offset+3] += 1.0
        V[j1*p + k0][rank_offset+3] += 1.0
        V[j0*p + k0][rank_offset+3] -= 1.0
        W[k0*n + i0][rank_offset+3] += 1.0
        W[k1*n + i0][rank_offset+3] += 1.0
        
        # m5
        U[i0*m + j0][rank_offset+4] += 1.0
        U[i0*m + j1][rank_offset+4] += 1.0
        V[j1*p + k1][rank_offset+4] += 1.0
        W[k0*n + i0][rank_offset+4] -= 1.0
        W[k1*n + i0][rank_offset+4] += 1.0
        
        # m6
        U[i1*m + j0][rank_offset+5] += 1.0
        U[i0*m + j0][rank_offset+5] -= 1.0
        V[j0*p + k0][rank_offset+5] += 1.0
        V[j0*p + k1][rank_offset+5] += 1.0
        W[k1*n + i1][rank_offset+5] += 1.0
        
        # m7
        U[i0*m + j1][rank_offset+6] += 1.0
        U[i1*m + j1][rank_offset+6] -= 1.0
        V[j1*p + k0][rank_offset+6] += 1.0
        V[j1*p + k1][rank_offset+6] += 1.0
        W[k0*n + i0][rank_offset+6] += 1.0
        
        return rank_offset + 7

    # Helper to add Schoolbook terms for a generic block
    def add_schoolbook_block(n_start, m_start, p_start, n_size, m_size, p_size, rank_offset):
        for li in range(n_size):
            for lj in range(m_size):
                for lk in range(p_size):
                    i = n_start + li
                    j = m_start + lj
                    k = p_start + lk
                    U[i*m + j][rank_offset] += 1.0
                    V[j*p + k][rank_offset] += 1.0
                    W[k*n + i][rank_offset] += 1.0
                    rank_offset += 1
        return rank_offset

    # Iterate through blocks
    for ni_idx, n_size in enumerate(n_sizes):
        n_start = n_offsets[ni_idx]
        for mi_idx, m_size in enumerate(m_sizes):
            m_start = m_offsets[mi_idx]
            for pi_idx, p_size in enumerate(p_sizes):
                p_start = p_offsets[pi_idx]
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    col_offset = add_strassen_block(n_start, m_start, p_start, col_offset)
                else:
                    col_offset = add_schoolbook_block(n_start, m_start, p_start, n_size, m_size, p_size, col_offset)
                
                if time.time() > deadline:
                    # Fallback if too slow (should not happen)
                    return _schoolbook_fallback(n, m, p)

    # Verification
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return _schoolbook_fallback(n, m, p)

def _verify(U, V, W, n, m, p, atol=1e-9):
    # Spot check a few random entries to avoid full O(n*m*p*rank) cost if possible, 
    # but for small dims it's fine.
    # Full check:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(len(U[0])):
                    val += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(val - 1.0) > atol:
                    return False
    return True

def _schoolbook_fallback(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
