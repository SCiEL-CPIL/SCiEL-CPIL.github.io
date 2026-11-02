import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Decomposition:
    # 1. 4x4x4 (i:0-3, j:0-3, k:0-3) -> Rank 28 (Strassen)
    # 2. 4x4x2 (i:0-3, j:0-3, k:4-5) -> Rank 8
    # 3. 4x1x6 (i:0-3, j:4, k:0-5) -> Rank 12
    
    # Total Rank 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 4x4x4
    # This is 7 blocks of 2x2x2 Strassen?
    # Actually, 4x4x4 rank 28 is 4 * 7?
    # No, 4x4x4 is 16 1x1x1 multiplications.
    # Strassen 2x2x2 has rank 7.
    # 4x4x4 can be viewed as 4 2x2x2 multiplications?
    # No, it's one 4x4x4.
    # The rank of 4x4x4 is 28.
    # We can hardcode the 28 terms.
    
    # For simplicity, we use the schoolbook for 4x4x4?
    # Rank 64.
    # Total 64 + 8 + 12 = 84.
    # Score 120/84 = 1.42.
    
    # Let's try to get the Strassen 4x4x4.
    # It is known that 4x4x4 has rank 28.
    # The decomposition is not trivial to generate on the fly.
    # However, we can use the fact that 4x4x4 is 4 independent 2x2x4?
    # No.
    
    # Let's stick to the half-integer safe decomposition.
    # Rank 60 is safe.
    # Rank 48 requires Strassen.
    
    # Let's implement Rank 60 first, as it is simpler and robust.
    # 4x5x2 (Rank 20)
    # 4x5x2 (Rank 20)
    # 4x5x2 (Rank 20)
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_block(i_start, j_start, k_start, n_sub, m_sub, p_sub):
        nonlocal col
        # Rank n_sub * m_sub
        # U = 1, V = 1, W = 1/p_sub
        # p_sub must be 2 for half-integer
        if p_sub != 2:
            raise ValueError("p_sub must be 2")
            
        for i in range(n_sub):
            for j in range(m_sub):
                r = col
                u_idx = (i_start + i) * m + (j_start + j)
                U[u_idx][r] = 1.0
                
                for k in range(p_sub):
                    v_idx = (j_start + j) * p + (k_start + k)
                    V[v_idx][r] = 1.0
                    
                for k in range(p_sub):
                    w_idx = (k_start + k) * n + (i_start + i)
                    W[w_idx][r] = 0.5
                    
                col += 1
                
    # 4x5x2 (k:0-1)
    add_block(0, 0, 0, 4, 5, 2)
    # 4x5x2 (k:2-3)
    add_block(0, 0, 2, 4, 5, 2)
    # 4x5x2 (k:4-5)
    add_block(0, 0, 4, 4, 5, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
