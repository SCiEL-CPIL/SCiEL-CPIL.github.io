import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 32)
    V = [[0] * rank for _ in range(m * p)]  # (32, 32)
    W = [[0] * rank for _ in range(p * n)]  # (16, 32)
    
    # Rank-32 decomposition for 2x4*8 matrix multiplication
    # Strategy: Split the 4 intermediate indices into two groups of 2.
    # Group 1: j=0,1 -> A[:,0:2] (2x2) * B[0:2,:] (2x8) -> C1 (2x8)
    # Group 2: j=2,3 -> A[:,2:4] (2x2) * B[2:4,:] (2x8) -> C2 (2x8)
    # C = C1 + C2
    #
    # For each 2x2*8 sub-problem, we use a rank-16 decomposition:
    # A (2x2) * B (2x8) where B = [B0 | B1 | B2 | B3] with Bk being 2x2.
    # C = A*B0 + A*B1 + A*B2 + A*B3
    # Each A*Bk is 2x2*2. The trivial rank for 2x2*2 is 4.
    # So 4 blocks * 4 terms = 16 terms per 2x2*8 sub-problem.
    # Total rank = 16 + 16 = 32.
    #
    # This is valid because the sub-problems are independent (different j ranges).
    # The trivial 4-term decomposition of 2x2*2 is:
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # Term for (i,j,k): u has 1 at (i,j), v has 1 at (j,k), w has 1 at (k,i)
    #
    # So for 2x2*8 with B split into 4 2x2 blocks:
    # For block b in 0..3 (columns 2b..2b+1 of B):
    #   For i in 0..1, j in 0..1, k in 0..1 (local k within block):
    #     global_k = 2*b + k
    #     u[i*2 + j] = 1
    #     v[j*8 + global_k] = 1
    #     w[global_k*2 + i] = 1
    #
    # Let's implement this.
    
    def add_2x2x8_block(j_offset, base_rank):
        """Add 16 rank-1 terms for a 2x2*8 sub-problem starting at intermediate index j_offset."""
        for b in range(4):  # 4 blocks of 2x2 in the 8 columns
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        r = base_rank + b * 4 + i * 2 + j * 2 + k
                        # U: index (i, j) in the 2x2 A block
                        # Global U index: i * m + (j_offset + j) = i * 4 + j_offset + j
                        U[i * 4 + (j_offset + j)][r] = 1
                        
                        # V: index (j_offset + j, 2*b + k) in B
                        # Global V index: (j_offset + j) * p + (2*b + k)
                        V[(j_offset + j) * 8 + (2 * b + k)][r] = 1
                        
                        # W: index (2*b + k, i) in C
                        # Global W index: (2*b + k) * n + i
                        W[(2 * b + k) * 2 + i][r] = 1
    
    # Block 1: j=0,1 (j_offset=0), uses ranks 0..15
    add_2x2x8_block(0, 0)
    
    # Block 2: j=2,3 (j_offset=2), uses ranks 16..31
    add_2x2x8_block(2, 16)
    
    return {"U": U, "V": V, "W": W}
