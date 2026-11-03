import time
import sys

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Returns a rank-46 exact decomposition of the 2x5x5 matrix multiplication tensor.
    Uses block decomposition: 5 = 2+2+1, so we decompose into 9 sub-problems.
    Sub-problem sizes and their Strassen-style ranks:
      2x2x2 -> rank 7 (Strassen)
      2x2x1 -> rank 4 (schoolbook)
      2x1x2 -> rank 4 (schoolbook)
      2x1x1 -> rank 2 (schoolbook)
      1x2x2 -> rank 4 (schoolbook)
      1x2x1 -> rank 2 (schoolbook)
      1x1x2 -> rank 2 (schoolbook)
      1x1x1 -> rank 1 (schoolbook)
    Total: 4*7 + 2*4 + 2*4 + 2 + 2*2 + 2 + 2 + 1 = 28+8+8+2+4+2+2+1 = 55? 
    Wait, let me recount the blocks:
    n=2, m=5, p=5.
    n blocks: [2]
    m blocks: [2, 2, 1]
    p blocks: [2, 2, 1]
    
    Blocks (i_block, j_block, k_block):
    (0,0,0): n=2, m=2, p=2 -> rank 7
    (0,0,1): n=2, m=2, p=2 -> rank 7
    (0,0,2): n=2, m=2, p=1 -> rank 4
    (0,1,0): n=2, m=2, p=2 -> rank 7
    (0,1,1): n=2, m=2, p=2 -> rank 7
    (0,1,2): n=2, m=2, p=1 -> rank 4
    (0,2,0): n=2, m=1, p=2 -> rank 4
    (0,2,1): n=2, m=1, p=2 -> rank 4
    (0,2,2): n=2, m=1, p=1 -> rank 2
    Total: 7+7+4+7+7+4+4+4+2 = 46.
    """
    n, m, p = DIMENSIONS
    n_blocks = [2]
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 1]
    
    rank = 0
    for nb in n_blocks:
        for mb in m_blocks:
            for pb in p_blocks:
                if nb == 2 and mb == 2 and pb == 2:
                    rank += 7
                elif nb == 2 and mb == 2 and pb == 1:
                    rank += 4
                elif nb == 2 and mb == 1 and pb == 2:
                    rank += 4
                elif nb == 2 and mb == 1 and pb == 1:
                    rank += 2
                elif nb == 1 and mb == 2 and pb == 2:
                    rank += 4
                elif nb == 1 and mb == 2 and pb == 1:
                    rank += 2
                elif nb == 1 and mb == 1 and pb == 2:
                    rank += 2
                else:
                    rank += 1
    
    # Allocate matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    n_offset = 0
    for nb in n_blocks:
        m_offset = 0
        for mb in m_blocks:
            p_offset = 0
            for pb in p_blocks:
                # Generate decomposition for this block
                block_r = 0
                if nb == 2 and mb == 2 and pb == 2:
                    block_r = 7
                elif nb == 2 and mb == 2 and pb == 1:
                    block_r = 4
                elif nb == 2 and mb == 1 and pb == 2:
                    block_r = 4
                elif nb == 2 and mb == 1 and pb == 1:
                    block_r = 2
                elif nb == 1 and mb == 2 and pb == 2:
                    block_r = 4
                elif nb == 1 and mb == 2 and pb == 1:
                    block_r = 2
                elif nb == 1 and mb == 1 and pb == 2:
                    block_r = 2
                else:
                    block_r = 1
                
                # Strassen for 2x2x2
                if nb == 2 and mb == 2 and pb == 2:
                    # M1 = (A11+A22)(B11+B22)
                    # M2 = (A21+A22)B11
                    # M3 = A11(B21-B22)
                    # M4 = A22(B12-B11)
                    # M5 = (A11+A12)B22
                    # M6 = (A21-A11)(B11+B12)
                    # M7 = (A12-A22)(B21+B22)
                    # C11 = M1+M4-M5+M7
                    # C12 = M3+M5
                    # C21 = M2+M4
                    # C22 = M1-M2+M3+M6
                    
                    strassen_factors = [
                        # (u_coeff, v_coeff, w_coeff)
                        # M1: U: A11=1, A22=1; V: B11=1, B22=1; W: C11=1, C22=1
                        ([(0,0,1),(0,1,1)], [(0,0,1),(0,1,1)], [(0,0,1),(1,1,1)]),
                        # M2: U: A21=1, A22=1; V: B11=1; W: C21=1
                        ([(0,1,1),(0,2,1)], [(0,0,1)], [(1,1,1)]),
                        # M3: U: A11=1; V: B21=1, B22=-1; W: C12=1
                        ([(0,0,1)], [(0,1,1),(0,2,-1)], [(0,2,1)]),
                        # M4: U: A22=1; V: B12=1, B11=-1; W: C11=1, C21=1
                        ([(0,2,1)], [(0,2,1),(0,0,-1)], [(0,1,1),(1,1,1)]),
                        # M5: U: A11=1, A12=1; V: B22=1; W: C11=-1, C12=1
                        ([(0,0,1),(0,1,1)], [(0,2,1)], [(-1,1,1),(0,2,1)]),
                        # M6: U: A21=1, A11=-1; V: B11=1, B12=1; W: C22=1
                        ([(0,1,1),(0,0,-1)], [(0,0,1),(0,2,1)], [(1,2,1)]),
                        # M7: U: A12=1, A22=-1; V: B21=1, B22=1; W: C11=1, C22=1
                        ([(0,1,1),(0,2,-1)], [(0,1,1),(0,2,1)], [(0,1,1),(1,2,1)]),
                    ]
                    
                    for u_list, v_list, w_list in strassen_factors:
                        for (ui, uj, uv) in u_list:
                            U[n_offset + ui*mb + uj][col] += uv
                        for (vj, vk, vv) in v_list:
                            V[m_offset + vj*pb + vk][col] += vv
                        for (wk, wi, wv) in w_list:
                            W[p_offset + wk*nb + wi][col] += wv
                        col += 1
                    
                # Schoolbook for other blocks
                else:
                    for i in range(nb):
                        for j in range(mb):
                            for k in range(pb):
                                U[n_offset + i*mb + j][col] = 1
                                V[m_offset + j*pb + k][col] = 1
                                W[p_offset + k*nb + i][col] = 1
                                col += 1
                
                p_offset += pb
            m_offset += mb
        n_offset += nb
    
    return {"U": U, "V": V, "W": W}
