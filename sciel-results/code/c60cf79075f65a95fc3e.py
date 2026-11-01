import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Try Winograd rank-7 per block (total rank 42)
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    # Winograd rank-7 decomposition for 2x2x2 all-ones tensor
    # U_block: 4x7, indexed by (i,j) -> i*2+j
    # V_block: 4x7, indexed by (j,k) -> j*2+k  
    # W_block: 4x7, indexed by (k,i) -> k*2+i
    
    # Standard Winograd F(2) x F(2) coefficients
    U_block = [
        [1,   0,   0,   0,   1,   1,   0],   # (i=0,j=0)
        [0,   1,   0,   0,   1,   0,   1],   # (i=0,j=1)
        [0,   0,   1,   0,   0,   1,   0],   # (i=1,j=0)
        [0,   0,   0,   1,   0,   0,   1],   # (i=1,j=1)
    ]
    
    V_block = [
        [1,   0,   1,   0,   1,   1,   0],   # (j=0,k=0)
        [0,   1,   0,   1,   1,   0,   1],   # (j=0,k=1)
        [1,   0,   1,   0,   1,   1,   0],   # (j=1,k=0)
        [0,   1,   0,   1,   1,   0,   1],   # (j=1,k=1)
    ]
    
    W_block = [
        [1,   0,   0,   0,  0.5,  0.5,  0.5],   # (k=0,i=0)
        [0,   1,   0,   0,  0.5,  0.5,  0.5],   # (k=0,i=1)
        [0,   0,   1,   0,  0.5,  0.5,  0.5],   # (k=1,i=0)
        [0,   0,   0,   1,  0.5,  0.5,  0.5],   # (k=1,i=1)
    ]
    
    # Verify the block decomposition
    U_b = np.array(U_block, dtype=float)
    V_b = np.array(V_block, dtype=float)
    W_b = np.array(W_block, dtype=float)
    
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for r in range(7):
                    val += U_b[i*2 + j][r] * V_b[j*2 + k][r] * W_b[k*2 + i][r]
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        # Build full matrices by tiling the block
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for block in range(num_blocks):
            offset = block * rank_per_block
            # U: rows are (i,j), same for all blocks
            for row in range(n * m):
                for r in range(rank_per_block):
                    U[row][offset + r] = U_block[row][r]
            
            # V: rows are (j,k), k = block*2 + k_local
            for j in range(m):
                for k_local in range(2):
                    row = j * p + block * 2 + k_local
                    v_row = j * 2 + k_local
                    for r in range(rank_per_block):
                        V[row][offset + r] = V_block[v_row][r]
            
            # W: rows are (k,i), k = block*2 + k_local
            for k_local in range(2):
                for i in range(n):
                    row = (block * 2 + k_local) * n + i
                    w_row = k_local * 2 + i
                    for r in range(rank_per_block):
                        W[row][offset + r] = W_block[w_row][r]
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook rank 48
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
