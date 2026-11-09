import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The directive suggests decomposing into 3 blocks of 1x5x5.
    # Rank of 1x5x5 is 25.
    # Total rank = 3 * 25 = 75.
    
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Each block corresponds to a row i of A
    for i in range(n):
        # For this block, we compute C[i, :] = A[i, :] * B
        # This is a 1x5x5 multiplication.
        # We use the schoolbook decomposition for this block.
        # Number of terms in this block is m * p = 5 * 5 = 25.
        block_rank = m * p
        rank += block_rank
        
        # Create zero matrices for this block's contribution
        U_block = np.zeros((n * m, block_rank), dtype=int)
        V_block = np.zeros((m * p, block_rank), dtype=int)
        W_block = np.zeros((p * n, block_rank), dtype=int)
        
        col = 0
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U_block[i * m + j, col] = 1
                
                # V index: j*p + k
                V_block[j * p + k, col] = 1
                
                # W index: k*n + i (as per prompt convention)
                W_block[k * n + i, col] = 1
                
                col += 1
        
        U_list.append(U_block)
        V_list.append(V_block)
        W_list.append(W_block)
        
    # Concatenate horizontally
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
