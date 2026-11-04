import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor M_{3,3,8} can be decomposed into disjoint column blocks.
    # We partition the p=8 columns into blocks.
    # For each block of width w, the sub-problem is M_{3,3,w}.
    # The rank of the total tensor is the sum of the ranks of the blocks.
    # Currently, we use the trivial decomposition for each block, which sums to 72.
    # This structure allows for easy replacement with better algorithms (e.g., Laderman for w=3).
    
    # Partition: 3 + 3 + 2
    blocks = [3, 3, 2]
    
    total_rank = 0
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    current_col_offset = 0
    
    for w in blocks:
        # Trivial decomposition for M_{3,3,w}
        # Rank is 3 * 3 * w
        rank_block = n * m * w
        
        U_b = np.zeros((nm, rank_block), dtype=np.float64)
        V_b = np.zeros((m * w, rank_block), dtype=np.float64) # Note: V dim depends on block width
        W_b = np.zeros((w * n, rank_block), dtype=np.float64) # Note: W dim depends on block width
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(w):
                    # A index (global)
                    idx_a = i * m + j
                    # B index (local to block, will be offset)
                    idx_b_local = j * w + k
                    # C index (local to block, will be offset)
                    idx_c_local = k * n + i
                    
                    U_b[idx_a, col] = 1.0
                    V_b[idx_b_local, col] = 1.0
                    W_b[idx_c_local, col] = 1.0
                    col += 1
        
        # We need to expand V_b and W_b to full size (mp and pn) by inserting zeros
        # and placing the block at the correct offset.
        
        V_full_block = np.zeros((mp, rank_block), dtype=np.float64)
        W_full_block = np.zeros((pn, rank_block), dtype=np.float64)
        
        # Place V_b into V_full_block
        # V_b has shape (m*w, rank_block)
        # The indices in B for this block are:
        # Row j (0..2) of B corresponds to rows j*w .. j*w+w-1 in the local block.
        # In the global B, these rows are offset by current_col_offset.
        # Global B index for (j, k) is j*p + (current_col_offset + k).
        # Local B index for (j, k) is j*w + k.
        
        for j in range(m):
            for k in range(w):
                local_idx = j * w + k
                global_idx = j * p + (current_col_offset + k)
                V_full_block[global_idx, :] = V_b[local_idx, :]
                
        # Place W_b into W_full_block
        # W_b has shape (w*n, rank_block)
        # W index convention: k*n + i.
        # Local k is 0..w-1.
        # Global k is current_col_offset + k.
        # Global W index for (k, i) is (current_col_offset + k)*n + i.
        # Local W index for (k, i) is k*n + i.
        
        for k in range(w):
            for i in range(n):
                local_idx = k * n + i
                global_idx = (current_col_offset + k) * n + i
                W_full_block[global_idx, :] = W_b[local_idx, :]
                
        U_blocks.append(U_b)
        V_blocks.append(V_full_block)
        W_blocks.append(W_full_block)
        
        total_rank += rank_block
        current_col_offset += w
        
    # Concatenate the blocks
    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    # Convert to lists
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
