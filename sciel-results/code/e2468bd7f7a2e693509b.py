import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin before 150s kill

    # Trivial decomposition: rank = n*m*p = 90
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1

    # Try to find a better decomposition using block decomposition
    # Strategy: split the indices into blocks and use known low-rank decompositions
    
    # For 3x5x6, we can split as:
    # I = {0,1,2}, J = {0,1,2,3,4}, K = {0,1,2,3,4,5}
    
    # One approach: split J into {0,1,2} and {3,4}
    # Then we have M_{3,3,6} and M_{3,2,6}
    # M_{3,3,6}: can further split K into {0,1,2} and {3,4,5}
    #   M_{3,3,3} (rank 23 with Lefterov) + M_{3,3,3} (rank 23) = 46
    # M_{3,2,6}: can split K into {0,1,2} and {3,4,5}
    #   M_{3,2,3} (rank 17) + M_{3,2,3} (rank 17) = 34
    # Total: 46 + 34 = 80
    
    # Alternatively, use the full trivial decomposition if we can't verify the blocks
    
    # Let's implement the block decomposition with verification
    # We'll build the U, V, W matrices block by block
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    total_rank = 0
    
    def add_block(i_indices, j_indices, k_indices, rank_func, U_block, V_block, W_block):
        """Add a block decomposition to the lists"""
        nonlocal total_rank
        n_block = len(i_indices)
        m_block = len(j_indices)
        p_block = len(k_indices)
        rank_block = rank_func(n_block, m_block, p_block)
        
        # Map local indices to global indices
        U_map = {}
        V_map = {}
        W_map = {}
        
        for li, gi in enumerate(i_indices):
            for lj in range(m_block):
                U_map[(li, lj)] = gi * m + j_indices[lj]
        
        for lj, gj in enumerate(j_indices):
            for lk in range(p_block):
                V_map[(lj, lk)] = gj * p + k_indices[lk]
        
        for lk, gk in enumerate(k_indices):
            for li in range(n_block):
                W_map[(lk, li)] = gk * n + i_indices[li]
        
        U_blocks.append((U_block, U_map, rank_block))
        V_blocks.append((V_block, V_map, rank_block))
        W_blocks.append((W_block, W_map, rank_block))
        total_rank += rank_block
    
    # Define rank functions for small cases
    def trivial_rank(n_b, m_b, p_b):
        return n_b * m_b * p_b
    
    def trivial_decomp(n_b, m_b, p_b):
        U = [[0] * (n_b * m_b * p_b) for _ in range(n_b * m_b)]
        V = [[0] * (n_b * m_b * p_b) for _ in range(m_b * p_b)]
        W = [[0] * (n_b * m_b * p_b) for _ in range(p_b * n_b)]
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i * m_b + j][col] = 1
                    V[j * p_b + k][col] = 1
                    W[k * n_b + i][col] = 1
                    col += 1
        return U, V, W, n_b * m_b * p_b
    
    # For now, use trivial decomposition for all blocks to ensure correctness
    # We can try to improve specific blocks later
    
    # Split: I={0,1,2}, J={0,1,2,3,4}, K={0,1,2,3,4,5}
    # Block 1: I={0,1,2}, J={0,1,2}, K={0,1,2,3,4,5}
    # Block 2: I={0,1,2}, J={3,4}, K={0,1,2,3,4,5}
    
    # For each block, use trivial decomposition
    # Block 1: 3x3x6, rank = 54
    # Block 2: 3x2x6, rank = 36
    # Total: 90 (same as trivial, no improvement)
    
    # To get improvement, we need to use non-trivial decompositions for sub-blocks
    # Let's try: split K in each block
    # Block 1a: I={0,1,2}, J={0,1,2}, K={0,1,2} -> 3x3x3, rank 23
    # Block 1b: I={0,1,2}, J={0,1,2}, K={3,4,5} -> 3x3x3, rank 23
    # Block 2a: I={0,1,2}, J={3,4}, K={0,1,2} -> 3x2x3, rank 17
    # Block 2b: I={0,1,2}, J={3,4}, K={3,4,5} -> 3x2x3, rank 17
    # Total: 23+23+17+17 = 80
    
    # We need explicit decompositions for 3x3x3 (rank 23) and 3x2x3 (rank 17)
    # For safety, if we can't verify, fall back to trivial
    
    # Let's implement and verify
    try:
        # 3x3x3 decomposition with rank 23 (Lefterov et al.)
        # This is complex; for now, use trivial for 3x3x3 (rank 27)
        # 3x2x3 decomposition with rank 17
        # For now, use trivial for 3x2x3 (rank 18)
        
        # Total with trivial sub-blocks: 27+27+18+18 = 90 (no improvement)
        # So we stick with the full trivial decomposition
        
        return {"U": U_trivial, "V": V_trivial, "W": W_trivial}
        
    except Exception as e:
        # Fall back to trivial
        return {"U": U_trivial, "V": V_trivial, "W": W_trivial}

# EVOLVE-BLOCK-END
