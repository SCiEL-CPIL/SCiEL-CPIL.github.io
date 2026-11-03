import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition
    # Split p into 4 blocks of size 2
    # For each block, split m into 2 and 1
    # <2, 2, 2> rank 4 (schoolbook)
    # <2, 1, 2> rank 4 (schoolbook)
    
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for k_block in range(4):
        k_start = k_block * 2
        
        # Part 1: <2, 2, 2>
        # A: rows 0,1; cols 0,1
        # B: rows 0,1; cols k_start, k_start+1
        # C: rows k_start, k_start+1; cols 0,1
        
        # Schoolbook for 2x2 * 2x2
        # Terms: (0,0), (0,1), (1,0), (1,1)
        # A[i,j] * B[j,k] -> C[i,k]
        
        # i=0, j=0, k=k_start
        U[0*3+0, idx] = 1.0 # A[0,0]
        V[0*8+k_start, idx] = 1.0 # B[0, k_start]
        W[k_start*2+0, idx] = 1.0 # C[k_start, 0]
        idx += 1
        
        # i=0, j=0, k=k_start+1
        U[0*3+0, idx] = 1.0
        V[0*8+k_start+1, idx] = 1.0
        W[k_start*2+1, idx] = 1.0
        idx += 1
        
        # i=0, j=1, k=k_start
        U[0*3+1, idx] = 1.0 # A[0,1]
        V[1*8+k_start, idx] = 1.0 # B[1, k_start]
        W[k_start*2+0, idx] = 1.0
        idx += 1
        
        # i=0, j=1, k=k_start+1
        U[0*3+1, idx] = 1.0
        V[1*8+k_start+1, idx] = 1.0
        W[k_start*2+1, idx] = 1.0
        idx += 1
        
        # i=1, j=0, k=k_start
        U[1*3+0, idx] = 1.0 # A[1,0]
        V[0*8+k_start, idx] = 1.0
        W[k_start*2+2, idx] = 1.0 # C[k_start, 1] -> index k*2+1? No.
        # W index is k*n + i.
        # k=k_start, i=1 -> k_start*2 + 1.
        # Wait, I used W[k_start*2+0] for i=0.
        # So for i=1, it is W[k_start*2+1].
        # Let's re-verify W index.
        # W shape (p*n, rank).
        # Index = k*n + i.
        # n=2.
        # k=k_start, i=0 -> 2*k_start.
        # k=k_start, i=1 -> 2*k_start + 1.
        # k=k_start+1, i=0 -> 2*(k_start+1).
        # k=k_start+1, i=1 -> 2*(k_start+1) + 1.
        
        # So my previous assignments:
        # i=0, k=k_start: W[2*k_start]
        # i=0, k=k_start+1: W[2*k_start+2]
        # i=1, k=k_start: W[2*k_start+1]
        # i=1, k=k_start+1: W[2*k_start+3]
        
        # I will rewrite the assignments carefully.
        
        # Reset idx to 0 and fill properly.
        
    # Let's restart the filling logic in the code below.
    idx = 0
    for k_block in range(4):
        k_start = k_block * 2
        
        # <2, 2, 2> part
        # A cols 0,1. B rows 0,1.
        # Terms (i, j, k_offset)
        # i in 0,1. j in 0,1. k_offset in 0,1.
        for i in range(2):
            for j in range(2):
                for k_off in range(2):
                    k_idx = k_start + k_off
                    # U index: i*m + j = i*3 + j
                    # V index: j*p + k_idx = j*8 + k_idx
                    # W index: k_idx*n + i = k_idx*2 + i
                    U[i*3 + j, idx] = 1.0
                    V[j*8 + k_idx, idx] = 1.0
                    W[k_idx*2 + i, idx] = 1.0
                    idx += 1
                    
        # <2, 1, 2> part
        # A col 2. B row 2.
        # i in 0,1. k_offset in 0,1.
        for i in range(2):
            for k_off in range(2):
                k_idx = k_start + k_off
                # U index: i*3 + 2
                # V index: 2*8 + k_idx = 16 + k_idx
                # W index: k_idx*2 + i
                U[i*3 + 2, idx] = 1.0
                V[16 + k_idx, idx] = 1.0
                W[k_idx*2 + i, idx] = 1.0
                idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
