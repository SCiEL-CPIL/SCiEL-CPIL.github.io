import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the 4x4x9 multiplication into:
    # 1. 4x4x4 multiplication (columns 0-3 of B)
    # 2. 4x4x4 multiplication (columns 4-7 of B)
    # 3. 4x4x1 multiplication (column 8 of B)
    
    # Total rank = 16 + 16 + 16 = 48
    
    rank = 48
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def add_trivial_block(u_offset, v_offset, w_offset, p_block):
        # Adds a trivial decomposition for A (4x4) * B (4 x p_block)
        # u_offset: index in U to start
        # v_offset: index in V to start
        # w_offset: index in W to start
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    u_idx = i * m + j
                    v_idx = j * p + v_offset + k
                    w_idx = (v_offset + k) * n + i
                    
                    U[u_idx, u_offset + col] = 1.0
                    V[v_idx, u_offset + col] = 1.0
                    W[w_idx, u_offset + col] = 1.0
                    col += 1
        return col
    
    # Block 1: B columns 0-3 (p_block=4)
    r1 = add_trivial_block(0, 0, 0, 4)
    
    # Block 2: B columns 4-7 (p_block=4)
    r2 = add_trivial_block(r1, 4, 4, 4)
    
    # Block 3: B column 8 (p_block=1)
    r3 = add_trivial_block(r1 + r2, 8, 8, 1)
    
    total_rank = r1 + r2 + r3
    assert total_rank == rank, f"Rank mismatch: {total_rank} vs {rank}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
