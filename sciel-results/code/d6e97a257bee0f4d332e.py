import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Partitions: [1, 4]
    # This allows using a 4x4x4 Strassen block (rank 49) instead of schoolbook (rank 64)
    # and small blocks for the rest.
    n_parts = [1, 4]
    m_parts = [1, 4]
    p_parts = [1, 4]
    
    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])
        
    # Calculate total rank
    total_rank = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                if ni == 4 and mj == 4 and pk == 4:
                    total_rank += 49 # Strassen 4x4x4
                else:
                    total_rank += ni * mj * pk
                    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_start, j_start, k_start, size, col_offset):
        """Recursively add Strassen factors for a size x size x size block."""
        nonlocal col
        if size == 1:
            # Schoolbook 1x1x1
            U[i_start * m + j_start][col] = 1
            V[j_start * p + k_start][col] = 1
            W[k_start * n + i_start][col] = 1
            col += 1
            return col_offset
        
        if size == 2:
            # Strassen 2x2x2
            i0, i1 = i_start, i_start + 1
            j0, j1 = j_start, j_start + 1
            k0, k1 = k_start, k_start + 1
            
            # M1
            U[i0 * m + j0][col] = 1
            U[i1 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            V[j1 * p + k1][col] = 1
            W[k0 * n + i0][col] = 1
            W[k1 * n + i1][col] = 1
            col += 1
            
            # M2
            U[i1 * m + j0][col] = 1
            U[i1 * m + j1][col] = 1
            V[j0 * p + k0][col] = 1
            W[k0 * n + i1][col] = 1
            W[k1 * n + i1][col] = -1
            col += 1
            
            # M3
            U[i0 * m + j0][col] = 1
            V[j0 * p + k1][col] = 1
            V[j1 * p + k1][col] = -1
            W[k1 * n + i0][col] = 1
            W[k1 * n + i1][col] = 1
            col += 1
            
            # M4
            U[i1 * m + j1][col] = 1
            V[j1 * p + k0][col] = 1
            V[j0 * p + k0][col] = -1
            W[k0 * n + i0][col] = 1
            W[k0 * n + i1][col] = 1
            col += 1
            
            # M5
            U[i0 * m + j0][col] = 1
            U[i0 * m + j1][col] = 1
            V[j1 * p + k1][col] = 1
            W[k0 * n + i0][col] = -1
            W[k1 * n + i0][col] = 1
            col += 1
            
            # M6
            U[i1 * m + j0][col] = 1
            U[i0 * m + j0][col] = -1
            V[j0 * p + k0][col] = 1
            V[j0 * p + k1][col] = 1
            W[k1 * n + i1][col] = 1
            col += 1
            
            # M7
            U[i0 * m + j1][col] = 1
            U[i1 * m + j1][col] = -1
            V[j1 * p + k0][col] = 1
            V[j1 * p + k1][col] = 1
            W[k0 * n + i0][col] = 1
            col += 1
            
            return col_offset

        # size > 2 (e.g., 4)
        half = size // 2
        # 8 sub-blocks
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    ni_start = i_start + di * half
                    nj_start = j_start + dj * half
                    nk_start = k_start + dk * half
                    col_offset = add_strassen_block(ni_start, nj_start, nk_start, half, col_offset)
        
        return col_offset

    for i in range(2):
        for j in range(2):
            for k in range(2):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                    
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 4 and mj == 4 and pk == 4:
                    col = add_strassen_block(i_start, j_start, k_start, 4, col)
                else:
                    # Schoolbook for other blocks
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
