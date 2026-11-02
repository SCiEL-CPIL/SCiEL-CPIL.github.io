import numpy as np
import time

def construct():
    _t0 = time.time()
    _deadline = _t0 + 115.0  # safety margin before 120s limit

    n, m, p = 2, 3, 8

    # Strategy: Split 3x8 into two 3x4 blocks
    # For each 3x4 block, split into two 3x2 sub-blocks
    # Each 3x2 sub-block uses rank-11 (Strassen 7 + 4 rank-1 extras)
    # Total: 4 sub-blocks * 11 = 44 rank

    # Actually, let's try a different decomposition:
    # Split the 8 columns of B into two groups of 4: B1 (cols 0-3) and B2 (cols 4-7)
    # For each 3x4 block, we can potentially use a more efficient scheme
    # 
    # Alternative: Treat as 2x3 * 3x8 = 2x8
    # Split output into 2x4 blocks
    # 
    # Let's stick with the block approach but try to optimize:
    # Split B into 4 blocks of 3x2, each using rank-11
    # Total rank = 44

    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44

    U = np.zeros((n * m, total_rank))  # (6, 44)
    V = np.zeros((m * p, total_rank))  # (24, 44)
    W = np.zeros((p * n, total_rank))  # (16, 44)

    for blk in range(num_blocks):
        base = blk * rank_per_block
        # For block blk, B columns are 2*blk and 2*blk+1
        # C columns (output) are 2*blk and 2*blk+1

        # Strassen for 2x2 core (A[:,0:2] * B[0:2,:]):
        # m1 = (A00+A11)(B00+B11)
        U[0, base + 0] = 1   # A[0,0]
        U[4, base + 0] = 1   # A[1,1]
        V[2 * blk, base + 0] = 1           # B[0, 2*blk]
        V[p + 2 * blk + 1, base + 0] = 1   # B[1, 2*blk+1]
        W[4 * blk, base + 0] = 1           # C[0, 2*blk]
        W[4 * blk + 3, base + 0] = 1       # C[1, 2*blk+1]

        # m2 = (A10+A11)B00
        U[3, base + 1] = 1   # A[1,0]
        U[4, base + 1] = 1   # A[1,1]
        V[2 * blk, base + 1] = 1           # B[0, 2*blk]
        W[4 * blk + 1, base + 1] = 1       # C[1, 2*blk]
        W[4 * blk + 3, base + 1] = -1      # C[1, 2*blk+1]

        # m3 = A00(B01-B11)
        U[0, base + 2] = 1   # A[0,0]
        V[2 * blk + 1, base + 2] = 1      # B[0, 2*blk+1]
        V[p + 2 * blk + 1, base + 2] = -1 # B[1, 2*blk+1]
        W[4 * blk + 2, base + 2] = 1      # C[0, 2*blk+1]
        W[4 * blk + 3, base + 2] = 1      # C[1, 2*blk+1]

        # m4 = A11(B10-B00)
        U[4, base + 3] = 1   # A[1,1]
        V[p + 2 * blk, base + 3] = 1     # B[1, 2*blk]
        V[2 * blk, base + 3] = -1        # B[0, 2*blk]
        W[4 * blk, base + 3] = 1         # C[0, 2*blk]
        W[4 * blk + 1, base + 3] = 1     # C[1, 2*blk]

        # m5 = (A00+A01)B11
        U[0, base + 4] = 1   # A[0,0]
        U[1, base + 4] = 1   # A[0,1]
        V[p + 2 * blk + 1, base + 4] = 1 # B[1, 2*blk+1]
        W[4 * blk, base + 4] = -1        # C[0, 2*blk]
        W[4 * blk + 2, base + 4] = 1     # C[0, 2*blk+1]

        # m6 = (A10-A00)(B00+B01)
        U[3, base + 5] = 1   # A[1,0]
        U[0, base + 5] = -1  # A[0,0]
        V[2 * blk, base + 5] = 1          # B[0, 2*blk]
        V[2 * blk + 1, base + 5] = 1      # B[0, 2*blk+1]
        W[4 * blk + 3, base + 5] = 1      # C[1, 2*blk+1]

        # m7 = (A01-A11)(B10+B11)
        U[1, base + 6] = 1   # A[0,1]
        U[4, base + 6] = -1  # A[1,1]
        V[p + 2 * blk, base + 6] = 1     # B[1, 2*blk]
        V[p + 2 * blk + 1, base + 6] = 1 # B[1, 2*blk+1]
        W[4 * blk, base + 6] = 1         # C[0, 2*blk]

        # Extra part: A[i,2] * B[2,k] for i in {0,1}, k in {2*blk, 2*blk+1}
        # 4 rank-1 terms
        U[2, base + 7] = 1
        V[2 * p + 2 * blk, base + 7] = 1
        W[4 * blk, base + 7] = 1

        U[2, base + 8] = 1
        V[2 * p + 2 * blk + 1, base + 8] = 1
        W[4 * blk + 2, base + 8] = 1

        U[5, base + 9] = 1
        V[2 * p + 2 * blk, base + 9] = 1
        W[4 * blk + 1, base + 9] = 1

        U[5, base + 10] = 1
        V[2 * p + 2 * blk + 1, base + 10] = 1
        W[4 * blk + 3, base + 10] = 1

    # Try to reduce rank by finding linear dependencies or using a better scheme
    # For now, return the rank-44 decomposition
    # Could potentially search for a lower rank decomposition within the time budget

    if time.time() > _deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
