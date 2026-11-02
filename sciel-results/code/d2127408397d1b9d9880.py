import numpy as np

def construct():
    n, m, p = 2, 3, 8
    # The problem decomposes into 4 independent 2x3x2 blocks.
    # Rank of M_{2,2,2} is 7 (Strassen).
    # Rank of M_{2,1,2} is 4 (Outer product of 2x1 and 1x2).
    # Since variable supports are disjoint, rank is additive: 7 + 4 = 11 per block.
    # Total rank = 4 * 11 = 44.
    
    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44
    
    U = np.zeros((n*m, total_rank))  # (6, 44)
    V = np.zeros((m*p, total_rank))  # (24, 44)
    W = np.zeros((p*n, total_rank))  # (16, 44)
    
    for blk in range(num_blocks):
        base = blk * rank_per_block
        # Block blk corresponds to B columns 2*blk, 2*blk+1 and C columns 2*blk, 2*blk+1
        
        # Indices for A: i*3+j (i in 0..1, j in 0..2)
        # Indices for B: j*p+k (j in 0..2, k in 0..7)
        # Indices for C: k*n+i (k in 0..7, i in 0..1)
        
        # --- Strassen 7 terms for the 2x2 core (A[:, 0:2] * B[0:2, 2*blk:2*blk+2]) ---
        # Let k0 = 2*blk, k1 = 2*blk+1
        
        # m1 = (A[0,0]+A[1,1])(B[0,k0]+B[1,k1])
        U[0*3+0, base+0] = 1
        U[1*3+1, base+0] = 1
        V[0*p + 2*blk, base+0] = 1
        V[1*p + (2*blk+1), base+0] = 1
        W[(2*blk)*n + 0, base+0] = 1
        W[(2*blk+1)*n + 1, base+0] = 1
        
        # m2 = (A[1,0]+A[1,1]) * B[0, k0]
        U[1*3+0, base+1] = 1
        U[1*3+1, base+1] = 1
        V[0*p + 2*blk, base+1] = 1
        W[(2*blk)*n + 1, base+1] = 1
        W[(2*blk+1)*n + 1, base+1] = -1
        
        # m3 = A[0,0] * (B[0, k1] - B[1, k1])
        U[0*3+0, base+2] = 1
        V[0*p + (2*blk+1), base+2] = 1
        V[1*p + (2*blk+1), base+2] = -1
        W[(2*blk+1)*n + 0, base+2] = 1
        W[(2*blk+1)*n + 1, base+2] = 1
        
        # m4 = A[1,1] * (B[1, k0] - B[0, k0])
        U[1*3+1, base+3] = 1
        V[1*p + 2*blk, base+3] = 1
        V[0*p + 2*blk, base+3] = -1
        W[(2*blk)*n + 0, base+3] = 1
        W[(2*blk)*n + 1, base+3] = 1
        
        # m5 = (A[0,0]+A[0,1]) * B[1, k1]
        U[0*3+0, base+4] = 1
        U[0*3+1, base+4] = 1
        V[1*p + (2*blk+1), base+4] = 1
        W[(2*blk)*n + 0, base+4] = -1
        W[(2*blk+1)*n + 0, base+4] = 1
        
        # m6 = (A[1,0]-A[0,0]) * (B[0, k0] + B[0, k1])
        U[1*3+0, base+5] = 1
        U[0*3+0, base+5] = -1
        V[0*p + 2*blk, base+5] = 1
        V[0*p + (2*blk+1), base+5] = 1
        W[(2*blk+1)*n + 1, base+5] = 1
        
        # m7 = (A[0,1]-A[1,1]) * (B[1, k0] + B[1, k1])
        U[0*3+1, base+6] = 1
        U[1*3+1, base+6] = -1
        V[1*p + 2*blk, base+6] = 1
        V[1*p + (2*blk+1), base+6] = 1
        W[(2*blk)*n + 0, base+6] = 1
        
        # --- 4 rank-1 terms for the extra column/row (A[:, 2] * B[2, k]) ---
        # C[0, k0] += A[0,2] * B[2, k0]
        U[0*3+2, base+7] = 1
        V[2*p + 2*blk, base+7] = 1
        W[(2*blk)*n + 0, base+7] = 1
        
        # C[0, k1] += A[0,2] * B[2, k1]
        U[0*3+2, base+8] = 1
        V[2*p + (2*blk+1), base+8] = 1
        W[(2*blk+1)*n + 0, base+8] = 1
        
        # C[1, k0] += A[1,2] * B[2, k0]
        U[1*3+2, base+9] = 1
        V[2*p + 2*blk, base+9] = 1
        W[(2*blk)*n + 1, base+9] = 1
        
        # C[1, k1] += A[1,2] * B[2, k1]
        U[1*3+2, base+10] = 1
        V[2*p + (2*blk+1), base+10] = 1
        W[(2*blk+1)*n + 1, base+10] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
