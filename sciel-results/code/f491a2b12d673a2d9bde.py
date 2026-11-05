import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    deadline = start_time + 110  # Safety margin

    # 2-level Strassen decomposition:
    # Level 1: 6x6 = 2x2 blocks of 3x3, use Strassen rank 7
    # Level 2: 3x3 = 2x2 blocks of 1.5x1.5, use Strassen rank 7
    # Total rank: 7 * 7 * ceil(1.5^3) = 7 * 7 * 4 = 196
    # But 1.5^3 = 3.375, we need 4 scalar terms per half-block
    # Actually for 1.5x1.5 matrices, naive rank = ceil(1.5)^3... 
    # Wait: 3x3 split into 2x2 blocks of size 1.5x1.5 doesn't work cleanly.
    # 
    # Alternative: 6x6 = 2x2 blocks of 3x3 (Strassen, rank 7)
    # Each 3x3: use schoolbook rank 27
    # Total: 7 * 27 = 189 (current)
    #
    # Better: 6x6 = 2x2 blocks of 3x3 (Strassen, rank 7)  
    # Each 3x3: split as 2x2 blocks of 1.5x1.5
    # For 1.5x1.5 "matrices" (which are really 1x1 scalar blocks with fractional indices),
    # we can't do Strassen directly. 
    #
    # Let me reconsider: 6 = 2*3, and 3 is odd, so we can't cleanly split 3x3 into 2x2 integer blocks.
    #
    # However, the task says "integer/half-integer entries" are allowed.
    # We can split 3x3 into overlapping blocks or use a different approach.
    #
    # Actually, let's try: 6x6 = 3x3 blocks of 2x2
    # Level 1: 3x3 outer, use schoolbook rank 27
    # Level 2: Each 2x2, use Strassen rank 7
    # Total: 27 * 7 = 189 (same as before, just different nesting)
    #
    # For improvement, we need a genuinely different decomposition.
    # Let me try: 6x6 split into 2x2 blocks of 3x3 (Strassen rank 7)
    # For each 3x3 inner: use a rank-21 algorithm (BCLR)
    # Total: 7 * 21 = 147
    #
    # BCLR rank-21 for 3x3: This is a known result.
    # The BCLR algorithm for 3x3 matrix multiplication has rank 21.
    # Let me implement this.

    # BCLR rank-21 decomposition for 3x3 matrix multiplication
    # Reference: Bini, Capovani, Lotti, Romani (1980)
    # The 21 terms for 3x3 multiplication
    
    # For 3x3 * 3x3 = 3x3, we need 21 bilinear forms
    # A[i,j] * B[j,k] -> C[i,k]
    
    # BCLR 21-term algorithm for 3x3:
    # This is complex; let me use a known formulation.
    
    # Actually, let me verify: the current 189 is already quite good.
    # 7 * 27 = 189. Can we do better?
    
    # Alternative: Use Strassen for 2x2 outer (rank 7), and for each 3x3 inner,
    # use a 21-term algorithm instead of 27-term schoolbook.
    # Total: 7 * 21 = 147
    
    # BCLR rank-21 for 3x3 matrix multiplication:
    # The 21 bilinear forms are:
    
    bclr_3x3 = [
        # Each entry: (A_indices, B_indices, C_indices)
        # A_indices: list of (i, j, coeff)
        # B_indices: list of (j, k, coeff)  
        # C_indices: list of (i, k, coeff)
        
        # Term 1: A[0,0]*B[0,0]
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        # Term 2: A[0,1]*B[1,0]
        ([(0,1,1)], [(1,0,1)], [(0,0,1)]),
        # Term 3: A[0,2]*B[2,0]
        ([(0,2,1)], [(2,0,1)], [(0,0,1)]),
        # Term 4: A[1,0]*B[0,1]
        ([(1,0,1)], [(0,1,1)], [(0,1,1)]),
        # Term 5: A[1,1]*B[1,1]
        ([(1,1,1)], [(1,1,1)], [(0,1,1)]),
        # Term 6: A[1,2]*B[2,1]
        ([(1,2,1)], [(2,1,1)], [(0,1,1)]),
        # Term 7: A[2,0]*B[0,2]
        ([(2,0,1)], [(0,2,1)], [(0,2,1)]),
        # Term 8: A[2,1]*B[1,2]
        ([(2,1,1)], [(1,2,1)], [(0,2,1)]),
        # Term 9: A[2,2]*B[2,2]
        ([(2,2,1)], [(2,2,1)], [(0,2,1)]),
        # Term 10: A[1,0]*B[0,0]
        ([(1,0,1)], [(0,0,1)], [(1,0,1)]),
        # Term 11: A[1,1]*B[1,0]
        ([(1,1,1)], [(1,0,1)], [(1,0,1)]),
        # Term 12: A[1,2]*B[2,0]
        ([(1,2,1)], [(2,0,1)], [(1,0,1)]),
        # Term 13: A[0,0]*B[0,1]
        ([(0,0,1)], [(0,1,1)], [(1,1,1)]),
        # Term 14: A[0,1]*B[1,1]
        ([(0,1,1)], [(1,1,1)], [(1,1,1)]),
        # Term 15: A[0,2]*B[2,1]
        ([(0,2,1)], [(2,1,1)], [(1,1,1)]),
        # Term 16: A[2,0]*B[0,0]
        ([(2,0,1)], [(0,0,1)], [(2,0,1)]),
        # Term 17: A[2,1]*B[1,0]
        ([(2,1,1)], [(1,0,1)], [(2,0,1)]),
        # Term 18: A[2,2]*B[2,0]
        ([(2,2,1)], [(2,0,1)], [(2,0,1)]),
        # Term 19: A[0,0]*B[0,2]
        ([(0,0,1)], [(0,2,1)], [(2,2,1)]),
        # Term 20: A[0,1]*B[1,2]
        ([(0,1,1)], [(1,2,1)], [(2,2,1)]),
        # Term 21: A[0,2]*B[2,2]
        ([(0,2,1)], [(2,2,1)], [(2,2,1)]),
    ]
    
    # Wait, that's just schoolbook 27 terms with 6 removed? No, that's still 21 terms but I need to verify correctness.
    # The above is NOT a valid rank-21 decomposition - it's missing cross terms.
    # 
    # Let me use the actual BCLR algorithm. The BCLR rank-21 for 3x3 is non-trivial.
    # 
    # For now, let me stick with the proven 189-rank solution but optimize the code.
    # The current solution is already at 189. Let me make sure it's correct and fast.
    
    block_size = 3
    num_blocks = 2
    dim = n  # 6
    
    rank_inner = block_size ** 3  # 27
    rank_outer = 7  # Strassen rank for 2x2
    total_rank = rank_inner * rank_outer  # 189
    
    # Initialize matrices
    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)
    
    # Strassen coefficients for 2x2 multiplication
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    bs = block_size
    
    for x_list, y_list, c_list in strassen_2x2:
        for r in range(bs):
            for s in range(bs):
                for c in range(bs):
                    for (ib, jb, x_coeff) in x_list:
                        u_idx = (ib * bs + r) * dim + (jb * bs + s)
                        U[u_idx, col] += x_coeff
                    
                    for (jb, kb, y_coeff) in y_list:
                        v_idx = (jb * bs + s) * dim + (kb * bs + c)
                        V[v_idx, col] += y_coeff
                    
                    for (ib, kb, c_coeff) in c_list:
                        w_idx = (kb * bs + c) * dim + (ib * bs + r)
                        W[w_idx, col] += c_coeff
                    
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
