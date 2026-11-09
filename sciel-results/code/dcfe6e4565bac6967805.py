"""Matrix multiplication 4x4x6 - rank-minimal tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 6
    
    # Strategy: Use a rank-22 decomposition for 4x4x4 core
    # and rank-12 for the 4x4x2 strip, total rank 34
    # 
    # The 4x4x4 tensor T[i,j,k] = 1 (for matrix mult) has border rank 21.
    # An explicit rank-22 decomposition exists. We use a known construction.
    #
    # For the 4x4x2 strip (i in [0,4), j in [0,4), k in [4,6)):
    # Each (i,j) pair needs 2 terms (for k=4,5), so 4*4*2 = 32 schoolbook
    # But we can do better: for fixed i, the strip is a 4x2 matrix mult
    # Actually the strip is: for each i in [0,4), j in [0,4), k in [4,6)
    # T[i,j,k] = 1, which is just 4*4*2 = 32 scalar products
    # But we can share: for fixed i, we need sum_j A[i,j]*B[j,4] and sum_j A[i,j]*B[j,5]
    # That's 4 * 2 = 8 terms per i, total 32. No sharing possible in general.
    #
    # Wait - let me reconsider. The full tensor for 4x4x6 has 4*4*6 = 96 entries.
    # Schoolbook rank = 96.
    #
    # Better approach: Use the structure. The 4x4x6 MM tensor can be decomposed as:
    # - 4x4x4 part (k=0..3): rank 22 (Landsberg)
    # - 4x4x2 part (k=4..5): rank 12 (using 2x2x2 Strassen on blocks, or just 8+4)
    #
    # For 4x4x2: We can view it as two 4x4x1 tensors.
    # A 4x4x1 tensor has rank 16 (schoolbook) but can be done in rank 8 using Strassen-like.
    # Actually 4x4x1 is just: C[i,0] = sum_j A[i,j] * B[j,0] for each i
    # That's 4 independent dot products of length 4, each rank 4, total rank 16.
    # But we can use Strassen: split into 2x2 blocks.
    # 2x2x1: C[i,0] = sum_j A[i,j]*B[j,0]. Rank of 2x2x1 tensor is 4 (schoolbook).
    # Can we do better? The 2x2x1 tensor has 4 entries, each needing one term.
    # Actually no sharing - each output entry depends on different (i,j) pairs.
    # So 2x2x1 has rank 4. Thus 4x4x2 has rank 4*4*2 = 32? No...
    #
    # Let me think more carefully. The tensor T[i,j,k] for i in [0,4), j in [0,4), k in [0,6)
    # has T[i,j,k] = 1 for all valid indices.
    #
    # A rank-r decomposition: T[i,j,k] = sum_{r=0}^{R-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    #
    # For the 4x4x4 part: known rank-22 decomposition exists.
    # For the 4x4x2 part: 
    #   For k=4: T[i,j,4] = 1 for all i,j. This is a 4x4x1 tensor.
    #   For k=5: T[i,j,5] = 1 for all i,j. This is another 4x4x1 tensor.
    #   The 4x4x1 tensor: T[i,j,0] = 1. Rank is 16 (each (i,j) needs its own term).
    #   But wait - can we share? T[i,j,0] = sum_r U[i,j,r]*V[j,0,r]*W[0,i,r]
    #   For this to equal 1 for all i,j, we need the sum to be 1.
    #   Minimum rank for 4x4x1 is 4 (use U[i,j,r] = delta_{i,r}, V[j,0,r] = 1, W[0,i,r] = delta_{j,r}... no)
    #   Actually: set U[i,j,r] = delta_{i,r}, V[j,0,r] = 1, W[0,i,r] = delta_{j,r}
    #   Then sum_r U[i,j,r]*V[j,0,r]*W[0,i,r] = sum_r delta_{i,r} * 1 * delta_{j,r} = delta_{i,j}
    #   That gives the identity, not 1. 
    #   
    #   For T[i,j,0] = 1: use U[i,j,r] = 1/sqrt(4), V[j,0,r] = 1, W[0,i,r] = 1 for all r
    #   Then sum = 4 * (1/2) * 1 * 1 = 2. Not quite.
    #   Use U[i,j,r] = 1/4, V[j,0,r] = 1, W[0,i,r] = 1, r=0..3: sum = 4*(1/4)*1*1 = 1. Yes!
    #   So 4x4x1 has rank 4!
    #   
    #   Similarly, 4x4x2: for k=4, use 4 terms; for k=5, use 4 more terms. Total rank 8.
    #   But we need to be careful: W[k*n+i, r] - for k=4, W[4*4+i, r] = W[16+i, r]
    #   For k=5, W[5*4+i, r] = W[20+i, r]
    #   These are different rows, so no conflict. Total rank for strip = 8.
    #
    # So total rank = 22 + 8 = 30? Let me verify the 4x4x4 rank-22.
    
    # Actually, let me use a simpler but still good approach:
    # Use 2x2x2 Strassen for the 4x4x4 part: 2*2*2 = 8 blocks, each rank 7, total 56.
    # Plus 4x4x2 strip with rank 8. Total = 64. Still better than 84.
    #
    # But can I do better? Let me try to use the fact that 4x4x4 has rank 22.
    # I'll hardcode a known rank-22 decomposition for 4x4x4.
    
    # Known rank-22 decomposition for 4x4x4 matrix multiplication:
    # This comes from the border rank 21 result, with explicit rank-22 construction.
    # Reference: Landsberg (1997), "On the rank and the border rank of tensor powers"
    #
    # I'll use a construction based on the 4x4x4 tensor.
    # The tensor T[i,j,k] = 1 for i,j,k in {0,1,2,3}
    #
    # A rank-22 construction:
    # We can use the fact that 4x4x4 can be decomposed using the "Coppersmith-Winograd" 
    # style or the explicit construction from the literature.
    #
    # Let me use a different approach: combine Strassen blocks with sharing.
    # 
    # Alternative: Use the 2x2x2 Strassen for 8 blocks (rank 56) but then try to 
    # find linear dependencies among the 56 columns to reduce rank.
    
    # Let me build the 56-rank decomposition first, then try to reduce it.
    
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # Build 4x4x4 part using Strassen: 8 blocks of 2x2x2, each rank 7 = 56
    n_blocks = 2
    m_blocks = 2
    p_blocks = 2
    
    # 4x4x2 strip: rank 8
    # For k=4: 4 terms (one per i), for k=5: 4 terms (one per i)
    
    total_rank_444 = 8 * 7  # 56
    total_rank_strip = 8
    total_rank = total_rank_444 + total_rank_strip  # 64
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j][col] = S_U[di * 2 + dj][c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k][col] = S_V[dj * 2 + dk][c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i][col] = S_W[dk * 2 + di][c]
                col_offset += 7
    
    # Now add the 4x4x2 strip (k=4,5) with rank 8
    # For k=4: T[i,j,4] = 1 for all i,j
    # Use: for each i in 0..3, one term with U[i,j,4+r] = 1/4, V[j,4,4+r] = 1, W[4*4+i, 4+r] = 1
    # Wait, that doesn't work because V[j,4,r] needs to be the same for all j.
    # 
    # Let me reconsider. For the strip T[i,j,k] = 1 for i in [0,4), j in [0,4), k in [4,6):
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k in strip
    #
    # For k=4: sum_r U[i*4+j, r] * V[j*6+4, r] * W[4*4+i, r] = 1
    # For k=5: sum_r U[i*4+j, r] * V[j*6+5, r] * W[5*4+i, r] = 1
    #
    # Simple construction: for k=4, use terms r=56..59 (4 terms):
    #   U[i*4+j, 56+s] = 1/4 for all i,j when s corresponds to i
    #   Actually: U[i*4+j, 56+s] = delta_{i,s} / 4 * 4 = delta_{i,s}
    #   V[j*6+4, 56+s] = 1
    #   W[4*4+i, 56+s] = delta_{j,s}
    #   Sum_s delta_{i,s} * 1 * delta_{j,s} = delta_{i,j}. That's not 1!
    #
    # I need sum to be 1, not delta_{i,j}.
    # 
    # Correct: U[i*4+j, 56+s] = 1/4, V[j*6+4, 56+s] = 1, W[4*4+i, 56+s] = 1 for s=0..3
    # Sum_s (1/4)*1*1 = 4*(1/4) = 1. Yes!
    #
    # Similarly for k=5: U[i*4+j, 60+s] = 1/4, V[j*6+5, 60+s] = 1, W[5*4+i, 60+s] = 1
    
    for s in range(4):
        r = 56 + s  # for k=4
        for i in range(4):
            for j in range(4):
                U[i * m + j][r] = 0.25
        for j in range(4):
            V[j * p + 4][r] = 1.0
        for i in range(4):
            W[4 * n + i][r] = 1.0
    
    for s in range(4):
        r = 60 + s  # for k=5
        for i in range(4):
            for j in range(4):
                U[i * m + j][r] = 0.25
        for j in range(4):
            V[j * p + 5][r] = 1.0
        for i in range(4):
            W[5 * n + i][r] = 1.0
    
    # Verify the decomposition
    import numpy as np
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_decomp(U_list, V_list, W_list):
        Ua = np.array(U_list, dtype=float)
        Va = np.array(V_list, dtype=float)
        Wa = np.array(W_list, dtype=float)
        rank = Ua.shape[1]
        
        T_decomp = np.zeros((n, m, p))
        for r in range(rank):
            u_col = Ua[:, r]
            v_col = Va[:, r]
            w_col = Wa[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        
        return np.allclose(T_decomp, 1.0, atol=1e-9)
    
    if not verify_decomp(U, V, W):
        # Fallback to pure Strassen 4x4x6
        total_rank_fb = 12 * 7  # 84
        U_fb = [[0] * total_rank_fb for _ in range(n * m)]
        V_fb = [[0] * total_rank_fb for _ in range(m * p)]
        W_fb = [[0] * total_rank_fb for _ in range(p * n)]
        
        col_offset = 0
        for bi in range(2):
            for bj in range(2):
                for bk in range(3):
                    for c in range(7):
                        col = col_offset + c
                        for di in range(2):
                            for dj in range(2):
                                i = bi * 2 + di
                                j = bj * 2 + dj
                                U_fb[i * m + j][col] = S_U[di * 2 + dj][c]
                        for dj in range(2):
                            for dk in range(2):
                                j = bj * 2 + dj
                                k = bk * 2 + dk
                                V_fb[j * p + k][col] = S_V[dj * 2 + dk][c]
                        for dk in range(2):
                            for di in range(2):
                                k = bk * 2 + dk
                                i = bi * 2 + di
                                W_fb[k * n + i][col] = S_W[dk * 2 + di][c]
                    col_offset += 7
        
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try to reduce rank by checking for linear dependencies
    # Build the "tensor matrix" and check if any column is in the span of others
    # This is expensive, so we'll do a quick check
    
    if time.time() > deadline - 20:
        return {"U": U, "V": V, "W": W}
    
    # Try greedy column elimination
    # For each column, check if removing it still gives a valid decomposition
    # This is O(R^2 * n*m*p) which is 64^2 * 96 ≈ 393K ops per check, 64 checks = 25M ops
    # Should be fast enough
    
    U_cur = [row[:] for row in U]
    V_cur = [row[:] for row in V]
    W_cur = [row[:] for row in W]
    cur_rank = total_rank
    
    # Try to remove columns from the end (strip columns first, then Strassen)
    for r_remove in range(cur_rank - 1, -1, -1):
        if time.time() > deadline - 10:
            break
        
        # Temporarily remove column r_remove
        U_test = [row[:r_remove] + row[r_remove+1:] for row in U_cur]
        V_test = [row[:r_remove] + row[r_remove+1:] for row in V_cur]
        W_test = [row[:r_remove] + row[r_remove+1:] for row in W_cur]
        
        if verify_decomp(U_test, V_test, W_test):
            U_cur = U_test
            V_cur = V_test
            W_cur = W_test
            cur_rank -= 1
    
    return {"U": U_cur, "V": V_cur, "W": W_cur}
# EVOLVE-BLOCK-END
