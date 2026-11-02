import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # We split the p=10 dimension into 5 blocks of size 2.
    # Each block is a 2x3x2 multiplication.
    # Rank of 2x3x2 is 8 (1 for outer product + 7 for Strassen).
    # Total rank = 5 * 8 = 40.
    
    num_blocks = p // 2
    block_p = 2
    rank_per_block = 8
    total_rank = num_blocks * rank_per_block
    
    # Initialize matrices
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Define the 8 terms for a 2x3x2 block
    # Local indices: i in {0,1}, j in {0,1,2}, k in {0,1}
    # U_idx = i * 3 + j
    # V_idx = j * 2 + k
    # W_idx = k * 2 + i
    
    # Term 0: Outer product for j=0
    # U: (0,0)=1, (1,0)=1 -> U_idx 0, 3
    # V: (0,0)=1, (0,1)=1 -> V_idx 0, 1
    # W: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1 -> W_idx 0, 1, 2, 3
    terms = [
        # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        (
            [0, 3], [1, 1],
            [0, 1], [1, 1],
            [0, 1, 2, 3], [1, 1, 1, 1]
        ),
        # Term 1: P1 = (A00+A11)(B00+B11)
        # A: i=0,j=1 (idx 1) val 1; i=1,j=2 (idx 5) val 1
        # B: j=1,k=0 (idx 2) val 1; j=2,k=1 (idx 5) val 1
        # C: i=0,k=0 (idx 0) val 1; i=1,k=1 (idx 3) val 1
        (
            [1, 5], [1, 1],
            [2, 5], [1, 1],
            [0, 3], [1, 1]
        ),
        # Term 2: P2 = (A10+A11)B00
        # A: i=1,j=1 (idx 4) val 1; i=1,j=2 (idx 5) val 1
        # B: j=1,k=0 (idx 2) val 1
        # C: i=1,k=0 (idx 2) val 1; i=1,k=1 (idx 3) val -1
        (
            [4, 5], [1, 1],
            [2], [1],
            [2, 3], [1, -1]
        ),
        # Term 3: P3 = A00(B01-B11)
        # A: i=0,j=1 (idx 1) val 1
        # B: j=1,k=1 (idx 3) val 1; j=2,k=1 (idx 5) val -1
        # C: i=0,k=0 (idx 0) val -1; i=0,k=1 (idx 1) val 1
        (
            [1], [1],
            [3, 5], [1, -1],
            [0, 1], [-1, 1]
        ),
        # Term 4: P4 = A11(B10-B00)
        # A: i=1,j=2 (idx 5) val 1
        # B: j=2,k=0 (idx 4) val 1; j=1,k=0 (idx 2) val -1
        # C: i=1,k=0 (idx 2) val 1; i=0,k=0 (idx 0) val 1
        (
            [5], [1],
            [4, 2], [1, -1],
            [2, 0], [1, 1]
        ),
        # Term 5: P5 = (A00+A01)B11
        # A: i=0,j=1 (idx 1) val 1; i=0,j=2 (idx 2) val 1
        # B: j=2,k=1 (idx 5) val 1
        # C: i=0,k=0 (idx 0) val -1; i=0,k=1 (idx 1) val 1
        (
            [1, 2], [1, 1],
            [5], [1],
            [0, 1], [-1, 1]
        ),
        # Term 6: P6 = (A10-A00)(B00+B01)
        # A: i=1,j=1 (idx 4) val 1; i=0,j=1 (idx 1) val -1
        # B: j=1,k=0 (idx 2) val 1; j=1,k=1 (idx 3) val 1
        # C: i=1,k=1 (idx 3) val 1
        (
            [4, 1], [1, -1],
            [2, 3], [1, 1],
            [3], [1]
        ),
        # Term 7: P7 = (A01-A11)(B10+B11)
        # A: i=0,j=2 (idx 2) val 1; i=1,j=2 (idx 5) val -1
        # B: j=2,k=0 (idx 4) val 1; j=2,k=1 (idx 5) val 1
        # C: i=0,k=0 (idx 0) val 1
        (
            [2, 5], [1, -1],
            [4, 5], [1, 1],
            [0], [1]
        ),
    ]
    
    for l in range(num_blocks):
        base_rank = l * rank_per_block
        for t_idx, (u_inds, u_vals, v_inds, v_vals, w_inds, w_vals) in enumerate(terms):
            r = base_rank + t_idx
            
            # Map local indices to global indices
            # U: local i*3+j is already global i*3+j because n=2, m=3
            # Global U index = i * m + j
            for u_ind, u_val in zip(u_inds, u_vals):
                U[u_ind, r] += u_val
            
            # V: local j*2+k. Global j*10 + (2*l + k)
            for v_ind, v_val in zip(v_inds, v_vals):
                j = v_ind // 2
                k = v_ind % 2
                global_v_ind = j * p + (2 * l + k)
                V[global_v_ind, r] += v_val
            
            # W: local k*2+i. Global (2*l + k)*2 + i
            for w_ind, w_val in zip(w_inds, w_vals):
                k = w_ind // 2
                i = w_ind % 2
                global_w_ind = (2 * l + k) * n + i
                W[global_w_ind, r] += w_val
                
    return {"U": U, "V": V, "W": W}
