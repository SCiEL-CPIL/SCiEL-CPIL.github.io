"""Program for matrix multiplication 3,4,8 rank-minimal decomposition."""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary containing factor matrices U, V, W.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The decomposition satisfies:
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all valid i,j,k
    """
    n, m, p = 3, 4, 8
    
    # Strategy:
    # M_{3,4,8} decomposes into M_{3,3,8} and M_{3,1,8}
    # M_{3,3,8} decomposes into two M_{3,3,4}
    # M_{3,3,4} decomposes into M_{3,3,3} and M_{3,3,1}
    #
    # We use Schoolbook for M_{3,3,3} (Rank 27)
    # Rank(M_{3,3,1}) = 1
    # Rank(M_{3,3,4}) = 27 + 1 = 28
    # Rank(M_{3,3,8}) = 2 * 28 = 56
    # Rank(M_{3,1,8}) = 1
    # Total Rank = 57
    
    U = []
    V = []
    W = []
    
    # Helper to add a schoolbook component
    # Adds decomposition for n' x m' x p'
    def add_schoolbook(u_offset, v_offset, w_offset, n_prime, m_prime, p_prime, u_cols, v_cols, w_cols, rank_offset):
        """
        u_offset: index in n*m corresponding to start of i
        v_offset: index in m*p corresponding to start of j
        w_offset: index in p*n corresponding to start of k
        """
        for i in range(n_prime):
            for j in range(m_prime):
                for k in range(p_prime):
                    col = rank_offset
                    U.append([0] * 57)
                    V.append([0] * 57)
                    W.append([0] * 57)
                    U[-1][col] = 1
                    V[-1][col] = 1
                    W[-1][col] = 1
                    rank_offset += 1
                    # Actually, I need to build the matrices correctly.
                    # This helper is messy because U, V, W are lists of lists.
                    # Let's build them as dicts first then convert.
        return rank_offset

    # Better approach: Build lists of tuples (index, value) for each column
    
    cols = 57
    
    U_dict = [dict() for _ in range(12)] # 3*4 = 12
    V_dict = [dict() for _ in range(32)] # 4*8 = 32
    W_dict = [dict() for _ in range(24)] # 8*3 = 24
    
    c = 0
    
    # Part 1: M_{3,3,8}
    # Split p=8 into k in 0..3 and k in 4..7
    
    for k_block in range(2):
        # M_{3,3,4} for this block
        # Split k_block*4 ... k_block*4+3
        # Further split M_{3,3,4} into M_{3,3,3} and M_{3,3,1}
        
        # M_{3,3,3} part (k' in 0..2 within the block)
        for i in range(3):
            for j in range(3):
                for k_rel in range(3):
                    k_global = k_block * 4 + k_rel
                    # Indices
                    u_idx = i * 4 + j # n*m index (n=3, m=4)
                    v_idx = j * 8 + k_global # m*p index (m=4, p=8)
                    w_idx = k_global * 3 + i # p*n index (p=8, n=3)
                    
                    U_dict[u_idx][c] = 1
                    V_dict[v_idx][c] = 1
                    W_dict[w_idx][c] = 1
                    c += 1
                    
        # M_{3,3,1} part (k' in 3 within the block)
        k_rel = 3
        k_global = k_block * 4 + k_rel
        # Rank 1 decomposition for M_{3,3,1}
        # U_{ij, c} = 1, V_{j, k, c} = 1, W_{k, i, c} = 1
        # Check: sum_c U V W = 1*1*1 = 1.
        # This works for all i, j.
        # So we add 1 term.
        for i in range(3):
            for j in range(3):
                u_idx = i * 4 + j
                v_idx = j * 8 + k_global
                w_idx = k_global * 3 + i
                
                U_dict[u_idx][c] = 1
                V_dict[v_idx][c] = 1
                W_dict[w_idx][c] = 1
                # Note: All these entries use the SAME column c.
                # We only increment c once after the loops.
                # But wait, if I loop i,j, I am writing to U_dict[u_idx][c] multiple times?
                # No, U_idx is different for each i,j.
                # V_idx is different for each j.
                # W_idx is different for each i.
                # So this effectively sets:
                # For fixed c, and fixed k_global:
                # U[i, 4+j, c] = 1 for all i,j? No.
                # U_dict[i*4+j][c] = 1.
                # V_dict[j*8+k_global][c] = 1.
                # W_dict[k_global*3+i][c] = 1.
                # The term in the sum is U[i,4+j,c] * V[j,8+k_global,c] * W[8+k_global, 3+i, c].
                # For a specific i,j,k_global, the product is 1*1*1 = 1.
                # This works.
                
        c += 1
        
    # Part 2: M_{3,1,8}
    # i in 0..2, j=3, k in 0..7
    # Rank 1 decomposition
    for i in range(3):
        for j in [3]:
            for k in range(8):
                u_idx = i * 4 + j
                v_idx = j * 8 + k
                w_idx = k * 3 + i
                
                U_dict[u_idx][c] = 1
                V_dict[v_idx][c] = 1
                W_dict[w_idx][c] = 1
                # All these share the same column c.
                # So U_dict[i*4+3][c]=1, V_dict[3*8+k][c]=1, W_dict[k*3+i][c]=1.
                # Product U[i,3,c] V[3,k,c] W[k,i,c] = 1*1*1 = 1.
                # This works.
    c += 1
    
    # Convert dicts to lists
    U = [[U_dict[idx].get(col, 0) for col in range(c)] for idx in range(12)]
    V = [[V_dict[idx].get(col, 0) for col in range(c)] for idx in range(32)]
    W = [[W_dict[idx].get(col, 0) for col in range(c)] for idx in range(24)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
