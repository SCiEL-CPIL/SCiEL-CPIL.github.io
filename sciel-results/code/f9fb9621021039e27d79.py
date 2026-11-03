import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define splits
    # n=3: [2, 1]
    # m=5: [2, 2, 1]
    # p=7: [2, 2, 2, 1]
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2, 1]
    
    # Pre-compute offsets
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
        
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
        
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
        
    U_full = []
    V_full = []
    W_full = []
    
    # Strassen factors for 2x2x2
    # U_s: 7x4, V_s: 7x4, W_s: 7x4
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U_s = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    V_s = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    W_s = [
        [1, 0, 0, 1], # M1: C00+1, C11+1
        [0, 1, 0, -1],# M2: C10+1, C11-1
        [0, 0, 1, 1], # M3: C01+1, C11+1
        [1, 1, 0, 0], # M4: C00+1, C10+1
        [-1, 0, 1, 0],# M5: C00-1, C01+1
        [0, 0, 0, 1], # M6: C11+1
        [1, 0, 0, 0]  # M7: C00+1
    ]
    
    for i_idx, ni in enumerate(n_splits):
        for j_idx, mj in enumerate(m_splits):
            for k_idx, pk in enumerate(p_splits):
                # Determine if we can use Strassen
                if ni == 2 and mj == 2 and pk == 2:
                    rank_b = 7
                    U_b = U_s
                    V_b = V_s
                    W_b = W_s
                else:
                    # Schoolbook
                    rank_b = ni * mj * pk
                    U_b = [[0]*rank_b for _ in range(ni*mj)]
                    V_b = [[0]*rank_b for _ in range(mj*pk)]
                    W_b = [[0]*rank_b for _ in range(pk*ni)]
                    
                    col = 0
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                U_b[ii*mj+jj][col] = 1
                                V_b[jj*pk+kk][col] = 1
                                W_b[kk*ni+ii][col] = 1
                                col += 1
                
                # Embed into full matrices
                # U row offset: i_idx offset * m + j_idx offset
                # Wait, U shape is (n*m, Rank).
                # Row index in full U: I*m + J.
                # I = n_offsets[i_idx] + ii.
                # J = m_offsets[j_idx] + jj.
                
                # We need to append rank_b columns to U_full, V_full, W_full
                # But U_full is a list of lists. We should build it column by column or row by row?
                # It's easier to build the full matrices as lists of rows, then append columns.
                # However, we are adding blocks. 
                # Let's initialize U_full, V_full, W_full as empty lists of rows? No, we don't know the final rank.
                # Let's collect blocks first.
                
                pass # Logic placeholder, will restructure below

    # Restructure: Collect all blocks, then assemble.
    blocks = []
    for i_idx, ni in enumerate(n_splits):
        for j_idx, mj in enumerate(m_splits):
            for k_idx, pk in enumerate(p_splits):
                if ni == 2 and mj == 2 and pk == 2:
                    blocks.append({
                        'type': 'strassen',
                        'i_idx': i_idx, 'j_idx': j_idx, 'k_idx': k_idx,
                        'ni': ni, 'mj': mj, 'pk': pk,
                        'U': U_s, 'V': V_s, 'W': W_s
                    })
                else:
                    rank_b = ni * mj * pk
                    U_b = [[0]*rank_b for _ in range(ni*mj)]
                    V_b = [[0]*rank_b for _ in range(mj*pk)]
                    W_b = [[0]*rank_b for _ in range(pk*ni)]
                    
                    col = 0
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                U_b[ii*mj+jj][col] = 1
                                V_b[jj*pk+kk][col] = 1
                                W_b[kk*ni+ii][col] = 1
                                col += 1
                    blocks.append({
                        'type': 'schoolbook',
                        'i_idx': i_idx, 'j_idx': j_idx, 'k_idx': k_idx,
                        'ni': ni, 'mj': mj, 'pk': pk,
                        'U': U_b, 'V': V_b, 'W': W_b
                    })
    
    # Calculate total rank
    total_rank = sum(len(b['U'][0]) for b in blocks)
    
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for b in blocks:
        rank_b = len(b['U'][0])
        ni = b['ni']
        mj = b['mj']
        pk = b['pk']
        
        n_off = n_offsets[b['i_idx']]
        m_off = m_offsets[b['j_idx']]
        p_off = p_offsets[b['k_idx']]
        
        for r in range(rank_b):
            col = col_offset + r
            
            # Map U
            for ii in range(ni):
                for jj in range(mj):
                    global_i = n_off + ii
                    global_j = m_off + jj
                    U_full[global_i*m + global_j][col] = b['U'][ii*mj+jj][r]
                    
            # Map V
            for jj in range(mj):
                for kk in range(pk):
                    global_j = m_off + jj
                    global_k = p_off + kk
                    V_full[global_j*p + global_k][col] = b['V'][jj*pk+kk][r]
                    
            # Map W
            for kk in range(pk):
                for ii in range(ni):
                    global_k = p_off + kk
                    global_i = n_off + ii
                    W_full[global_k*n + global_i][col] = b['W'][kk*ni+ii][r]
                    
        col_offset += rank_b
        
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
