"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Partitioning strategy:
    # n splits: [2, 2]
    # m splits: [2, 2, 1]
    # p splits: [2, 2, 2]
    
    n_splits = [2, 2]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    # Calculate offsets
    n_offsets = []
    curr = 0
    for size in n_splits:
        n_offsets.append(curr)
        curr += size
    m_offsets = []
    curr = 0
    for size in m_splits:
        m_offsets.append(curr)
        curr += size
    p_offsets = []
    curr = 0
    for size in p_splits:
        p_offsets.append(curr)
        curr += size
        
    # Determine total rank
    total_rank = 0
    block_ranks = []
    for ni in range(len(n_splits)):
        for mi in range(len(m_splits)):
            for pi in range(len(p_splits)):
                nb = n_splits[ni]
                mb = m_splits[mi]
                pb = p_splits[pi]
                if nb == 2 and mb == 2 and pb == 2:
                    block_ranks.append(7)
                else:
                    block_ranks.append(nb * mb * pb)
                total_rank += block_ranks[-1]
                
    # Initialize global matrices
    U_full = [[0] * total_rank for _ in range(nm)]
    V_full = [[0] * total_rank for _ in range(mp)]
    W_full = [[0] * total_rank for _ in range(pn)]
    
    col = 0
    for ni in range(len(n_splits)):
        for mi in range(len(m_splits)):
            for pi in range(len(p_splits)):
                nb = n_splits[ni]
                mb = m_splits[mi]
                pb = p_splits[pi]
                
                n_off = n_offsets[ni]
                m_off = m_offsets[mi]
                p_off = p_offsets[pi]
                
                rank_b = block_ranks[ni * len(m_splits) * len(p_splits) + mi * len(p_splits) + pi]
                
                if nb == 2 and mb == 2 and pb == 2:
                    # Strassen 2x2x2
                    # Local U: 4x7, V: 4x7, W: 4x7
                    U_b = [[0]*7 for _ in range(4)]
                    V_b = [[0]*7 for _ in range(4)]
                    W_b = [[0]*7 for _ in range(4)]
                    # M1=(A11+A22)(B11+B22): C11+=M1, C22+=M1
                    U_b[0][0]=1; U_b[3][0]=1; V_b[0][0]=1; V_b[3][0]=1; W_b[0][0]=1; W_b[3][0]=1
                    # M2=(A21+A22)B21: C21+=M2
                    U_b[2][1]=1; U_b[3][1]=1; V_b[2][1]=1; W_b[2][1]=1
                    # M3=A11(B12-B22): C12+=M3
                    U_b[0][2]=1; V_b[1][2]=1; V_b[3][2]=-1; W_b[1][2]=1
                    # M4=A22(B21-B11): C11+=M4, C21+=M4
                    U_b[3][3]=1; V_b[2][3]=1; V_b[0][3]=-1; W_b[0][3]=1; W_b[2][3]=1
                    # M5=(A11+A12)B22: C11-=M5, C12+=M5
                    U_b[0][4]=1; U_b[1][4]=1; V_b[3][4]=1; W_b[0][4]=-1; W_b[1][4]=1
                    # M6=(A21-A11)(B11+B12): C22+=M6
                    U_b[2][5]=1; U_b[0][5]=-1; V_b[0][5]=1; V_b[1][5]=1; W_b[3][5]=1
                    # M7=(A12-A22)(B21+B22): C11+=M7, C22+=M7
                    U_b[1][6]=1; U_b[3][6]=-1; V_b[2][6]=1; V_b[3][6]=1; W_b[0][6]=1; W_b[3][6]=1
                    
                    # Map to global
                    for i_loc in range(nb):
                        for j_loc in range(mb):
                            g_i = n_off + i_loc
                            g_j = m_off + j_loc
                            g_u_row = g_i * m + g_j
                            for r in range(7):
                                U_full[g_u_row][col + r] = U_b[i_loc * mb + j_loc][r]
                    
                    for j_loc in range(mb):
                        for k_loc in range(pb):
                            g_j = m_off + j_loc
                            g_k = p_off + k_loc
                            g_v_row = g_j * p + g_k
                            for r in range(7):
                                V_full[g_v_row][col + r] = V_b[j_loc * pb + k_loc][r]
                    
                    for k_loc in range(pb):
                        for i_loc in range(nb):
                            g_k = p_off + k_loc
                            g_i = n_off + i_loc
                            g_w_row = g_k * n + g_i
                            for r in range(7):
                                W_full[g_w_row][col + r] = W_b[k_loc * nb + i_loc][r]
                    
                    col += 7
                else:
                    # Schoolbook
                    rank_b = nb * mb * pb
                    for i_loc in range(nb):
                        for j_loc in range(mb):
                            for k_loc in range(pb):
                                g_i = n_off + i_loc
                                g_j = m_off + j_loc
                                g_k = p_off + k_loc
                                
                                g_u_row = g_i * m + g_j
                                g_v_row = g_j * p + g_k
                                g_w_row = g_k * n + g_i
                                
                                U_full[g_u_row][col] = 1
                                V_full[g_v_row][col] = 1
                                W_full[g_w_row][col] = 1
                                col += 1
                                
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U_full[i*m+j][r] * V_full[j*p+k][r] * W_full[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
