import numpy as np

def construct():
    n, m, p = 4, 4, 6
    
    def get_strassen(n):
        if n == 1:
            return [[1]], [[1]], [[1]]
        
        m_sub = n // 2
        U_sub, V_sub, W_sub = get_strassen(m_sub)
        rank_sub = len(U_sub)
        rank_full = 7 * rank_sub
        
        U = [[0] * rank_full for _ in range(n * n)]
        V = [[0] * rank_full for _ in range(n * n)]
        W = [[0] * rank_full for _ in range(n * n)]
        
        def add_sub(U, V, W, A_blocks, B_blocks, C_blocks, signs, offset):
            for k in range(rank_sub):
                for (r, c) in A_blocks:
                    for i in range(m_sub):
                        for j in range(m_sub):
                            U[(r + i) * n + (c + j)][offset + k] += U_sub[i * m_sub + j][k]
                
                for (r, c) in B_blocks:
                    for i in range(m_sub):
                        for j in range(m_sub):
                            V[(r + i) * n + (c + j)][offset + k] += V_sub[i * m_sub + j][k]
                
                for idx, (r, c) in enumerate(C_blocks):
                    sign = signs[idx]
                    for i in range(m_sub):
                        for j in range(m_sub):
                            W[(r + i) * n + (c + j)][offset + k] += sign * W_sub[i * m_sub + j][k]
        
        add_sub(U, V, W, [(0,0), (m_sub,m_sub)], [(0,0), (m_sub,m_sub)], 
                [(0,0), (0,m_sub), (m_sub,0), (m_sub,m_sub)], [1, -1, -1, 1], 0)
        
        add_sub(U, V, W, [(m_sub,0), (m_sub,m_sub)], [(0,0)], 
                [(m_sub,0)], [1], rank_sub)
        
        add_sub(U, V, W, [(0,0)], [(0,m_sub), (m_sub,m_sub)], 
                [(0,m_sub)], [1, -1], 2 * rank_sub)
        
        add_sub(U, V, W, [(m_sub,m_sub)], [(m_sub,0), (0,0)], 
                [(m_sub,m_sub)], [1, -1], 3 * rank_sub)
        
        add_sub(U, V, W, [(0,0), (0,m_sub)], [(m_sub,m_sub)], 
                [(0,0)], [1, 1], 4 * rank_sub)
        
        add_sub(U, V, W, [(m_sub,0), (0,0)], [(0,0), (0,m_sub)], 
                [(m_sub,0), (m_sub,m_sub)], [1, 1, -1, -1], 5 * rank_sub)
        
        add_sub(U, V, W, [(0,m_sub), (m_sub,m_sub)], [(m_sub,0), (m_sub,m_sub)], 
                [(0,m_sub), (m_sub,m_sub)], [1, -1, 1, 1], 6 * rank_sub)
        
        return U, V, W

    U4, V4, W4 = get_strassen(4)
    
    rank_4 = len(U4)
    rank_2 = 4 * 4 * 2
    total_rank = rank_4 + rank_2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy Strassen U
    for i in range(16):
        for k in range(rank_4):
            U[i][k] = U4[i][k]
            
    # Copy Strassen V with index remapping
    for i in range(16):
        j = i // 4
        k = i % 4
        for r in range(rank_4):
            V[j * 6 + k][r] = V4[i][r]
            
    # Copy Strassen W
    for i in range(16):
        for k in range(rank_4):
            W[i][k] = W4[i][k]
            
    # Add Schoolbook for B columns 4, 5
    col = rank_4
    for i in range(4):
        for j in range(4):
            for k in range(2):
                b_col = 4 + k
                U[i * 4 + j][col] = 1
                V[j * 6 + b_col][col] = 1
                W[b_col * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
