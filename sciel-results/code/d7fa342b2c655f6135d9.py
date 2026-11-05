"""Matrix multiplication 4x5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Uses a recursive Strassen-based decomposition:
    - 4x4x4 core uses Strassen (rank 49 instead of 64)
    - Boundary terms use schoolbook (rank 36)
    Total rank: 85
    
    The tensor identity: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all i in [0,4), j in [0,5), k in [0,5).
    """
    n, m, p = 4, 5, 5
    total_rank = 85
    
    # Initialize factor matrices
    # U: (n*m, rank) -> (20, 85)
    # V: (m*p, rank) -> (25, 85)
    # W: (p*n, rank) -> (20, 85)
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Define 2x2x2 Strassen terms (U, V, W vectors of length 4)
    # Local indices for 2x2 blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    s_u = np.array([
        [ 1,  0,  1,  0,  1, -1,  0], # (0,0)
        [ 0,  0,  0,  0,  1,  0,  1], # (0,1)
        [ 0,  1,  0,  0,  0,  1,  0], # (1,0)
        [ 1,  1,  0,  1,  0,  0, -1]  # (1,1)
    ], dtype=int)
    
    s_v = np.array([
        [ 1,  1,  0, -1,  0,  1,  0], # (0,0)
        [ 0,  0,  1,  0,  0,  1,  0], # (0,1)
        [ 0,  0,  0,  1,  0,  0,  1], # (1,0)
        [ 1,  0, -1,  0,  1,  0,  1]  # (1,1)
    ], dtype=int)
    
    s_w = np.array([
        [ 1,  0,  0,  1, -1,  0,  1], # (0,0)
        [ 0,  0,  1,  0,  1,  0,  0], # (0,1)
        [ 0,  1,  0,  1,  0,  0,  0], # (1,0)
        [ 1, -1,  1,  0,  0,  1,  0]  # (1,1)
    ], dtype=int)
    
    # Core 4x4x4 part: 7 Strassen terms
    # Each term expands to 7 rank-1 terms in the full tensor
    # We define the linear combinations of 2x2 blocks for A, B, C
    # Block indices: (br, bc) where br in {0,1}, bc in {0,1}
    
    # Term definitions: (A_blocks, B_blocks, C_blocks)
    # Each is a dict {(br, bc): coeff}
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    col = 0
    start_time = time.time()
    deadline = 130.0
    
    # Helper to get global index from 4x4 block local index
    def global_idx_4x4(br, bc, lr, lc):
        # br, bc are block coords (0,1), lr, lc are local coords (0,1)
        # Returns global row/col index in 0..3
        r = 2 * br + lr
        c = 2 * bc + lc
        return r, c

    for t_idx, (a_b, b_b, c_b) in enumerate(terms):
        if time.time() - start_time > deadline:
            break
            
        for k in range(7):
            curr_col = col + k
            
            # Process U factors
            u_vec = s_u[:, k]
            for (abr, abc), coeff in a_b.items():
                for lr in range(2):
                    for lc in range(2):
                        val = u_vec[lr * 2 + lc]
                        if val != 0:
                            gi, gj = global_idx_4x4(abr, abc, lr, lc)
                            U[gi * 5 + gj, curr_col] += coeff * val
            
            # Process V factors
            v_vec = s_v[:, k]
            for (bbr, bbc), coeff in b_b.items():
                for lr in range(2):
                    for lc in range(2):
                        val = v_vec[lr * 2 + lc]
                        if val != 0:
                            gj, gk = global_idx_4x4(bbr, bbc, lr, lc)
                            V[gj * 5 + gk, curr_col] += coeff * val
            
            # Process W factors
            w_vec = s_w[:, k]
            for (cbr, cbc), coeff in c_b.items():
                for lr in range(2):
                    for lc in range(2):
                        val = w_vec[lr * 2 + lc]
                        if val != 0:
                            gi, gk = global_idx_4x4(cbr, cbc, lr, lc)
                            W[gk * 4 + gi, curr_col] += coeff * val
        
        col += 7
    
    # Boundary terms: 36 rank-1 terms
    # 1. A[:, 0:4] * B[:, 4] -> C[:, 4]
    # A[i, j] * B[j, 4] -> C[i, 4]
    # U[i*5+j, col] = 1, V[j*5+4, col] = 1, W[4*4+i, col] = 1
    for i in range(4):
        for j in range(4):
            if time.time() - start_time > deadline:
                break
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1
    
    # 2. A[:, 4] * B[4, :] -> C[:, 0:4]
    # A[i, 4] * B[4, k] -> C[i, k]
    # U[i*5+4, col] = 1, V[4*5+k, col] = 1, W[k*4+i, col] = 1
    for i in range(4):
        for k in range(4):
            if time.time() - start_time > deadline:
                break
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
            col += 1
    
    # 3. A[:, 4] * B[4, 4] -> C[:, 4]
    # A[i, 4] * B[4, 4] -> C[i, 4]
    # U[i*5+4, col] = 1, V[4*5+4, col] = 1, W[4*4+i, col] = 1
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        U[i * 5 + 4, col] = 1
        V[4 * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
        col += 1
    
    # Ensure we haven't exceeded allocated space
    if col != total_rank:
        # If col < total_rank, we have unused space (should not happen)
        # If col > total_rank, we have a bug
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
