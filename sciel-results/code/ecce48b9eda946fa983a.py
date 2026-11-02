"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_exact(U, V, W, n, m, p):
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _bcr_block_decomposition():
    """
    Apply BCR block method: partition 4x4x9 into 2x2x3 blocks.
    4 = 2*2, 4 = 2*2, 9 = 3*3
    So we have 2*2*3 = 12 blocks, each 2x2x3.
    
    For a 2x2x3 matrix multiplication, we can use a rank-7 decomposition.
    Total rank = 12 * 7 = 84.
    
    The 2x2x3 tensor: C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    
    We need to find U (4 x 7), V (6 x 7), W (6 x 7) such that:
    sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = delta for each (i,j,k)
    """
    n, m, p = 4, 4, 9
    num_ib = n // 2  # 2
    num_jb = m // 2  # 2
    num_kb = p // 3  # 3
    rank_per_block = 7
    total_rank = num_ib * num_jb * num_kb * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # For each 2x2x3 block, define the rank-7 decomposition
    # The 2x2x3 multiplication tensor has entries T[i,j,k] = 1 for all i,j,k
    # where i in {0,1}, j in {0,1}, k in {0,1,2}
    # 
    # A simple rank-7 decomposition for 2x2x3:
    # We can use the fact that 2x2x2 has rank 7 (Winograd), and add one more term for the 3rd k dimension.
    # Actually, let me think more carefully.
    #
    # For 2x2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # The tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
    #
    # One approach: use rank 8 (schoolbook for 2x2x3) and then try to reduce.
    # But let's use a known decomposition.
    #
    # Actually, the rank of 2x2x3 is known to be 7 (same as 2x2x2 since the third k dimension
    # can be handled with the same structure).
    #
    # Let me construct it explicitly:
    # For 2x2x2, Winograd's rank-7 decomposition:
    # U (4x7), V (4x7), W (4x7)
    #
    # For 2x2x3, we can extend this. Let me use a simple construction:
    # Terms 0-6: same as 2x2x2 Winograd for k=0,1
    # Term 7: for k=2
    #
    # But that gives rank 8. Let me try a different approach.
    #
    # Actually, I'll use a rank-8 decomposition for simplicity and correctness,
    # which gives total rank = 12 * 8 = 96.
    # This is still better than schoolbook 144.
    
    rank_per_block = 8
    total_rank = num_ib * num_jb * num_kb * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ib in range(num_ib):
        for jb in range(num_jb):
            for kb in range(num_kb):
                # Block (ib, jb, kb):
                # A rows: 2*ib, 2*ib+1
                # B cols: 2*jb, 2*jb+1
                # C cols: 3*kb, 3*kb+1, 3*kb+2
                
                # For this 2x2x3 block, use rank-8 schoolbook:
                # 2*2*3 = 12 terms, but we can do better.
                # Actually, let me just use the schoolbook for each block:
                # 2*2*3 = 12 terms per block, total 12*12 = 144 (same as overall schoolbook)
                #
                # Let me use a rank-7 decomposition for 2x2x3.
                #
                # Here's a rank-7 decomposition for 2x2x3:
                # We identify the 2x2x3 tensor with the 2x2x2 tensor plus an extra slice.
                #
                # Actually, I realize I should just use a known construction.
                # Let me use the following rank-8 decomposition for 2x2x3:
                # Terms 0-3: A[i,0]*B[0,k] for i in {0,1}, k in {0,1,2} - that's 6 terms
                # Terms 4-5: A[i,1]*B[1,k] for i in {0,1}, k in {0,1,2} - that's 6 terms
                # Total: 12 terms (schoolbook)
                #
                # To get rank < 12, I need a smarter decomposition.
                #
                # Let me just use the overall schoolbook for now, since getting a valid
                # rank-7 decomposition for 2x2x3 requires careful construction.
                pass
    
    # Since constructing a valid rank-7 decomposition for 2x2x3 is non-trivial,
    # let me fall back to schoolbook but with a different approach.
    # Actually, let me just return the schoolbook decomposition.
    return _schoolbook()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    best = _schoolbook()
    return best
# EVOLVE-BLOCK-END
