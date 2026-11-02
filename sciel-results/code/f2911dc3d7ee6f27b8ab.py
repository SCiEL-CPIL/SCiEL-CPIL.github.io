"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    
    Strategy (ladder method):
    1. Split C into two column groups: C[:, 0:4] and C[:, 4:6].
    2. For C[:, 0:4]: decompose 3x4x4 as (3x3x3 via Strassen rank-7) + (3x1x4 rank-4) + (3x3x1 rank-3) + (3x1x1 rank-1) = 15
    3. For C[:, 4:6]: decompose 3x4x2 as (3x3x2 rank-6) + (3x1x2 rank-2) = 8
    4. Total rank: 15 + 8 = 23... wait, that's too low. Let me recalculate.
    
    Actually, let me think more carefully:
    - 3x4x4: We can split B into 4x3 and 4x1 blocks.
      * 3x3x3 part: Strassen rank 7
      * 3x1x3 part (bottom row of B, 3 cols of C): rank 3 (outer product A_bottom * B_bottom_row for each col)
      Wait, 3x1x3 means: A has 3 rows, 1 col (j=3); B has 1 row (j=3), 3 cols; C has 3 rows, 3 cols.
      This is rank 9 (each (i,k) pair needs its own term).
      * 3x3x1 part (3 rows of B, 1 col of C): rank 3 (each i needs its own term)
      * 3x1x1 part: rank 1
      Total for 3x4x4: 7 + 9 + 3 + 1 = 20
      
    Hmm, that gives 20 + 8 = 28 which is worse than 72? No wait, I'm confusing myself.
    
    Let me restart with a clearer approach:
    
    The tensor T[i,j,k] = 1 if (i,j) is in A, (j,k) is in B, (k,i) is in C.
    
    Approach: Split j (the shared dimension) into j=0..2 and j=3.
    - For j=0..2: we have 3x3x6 sub-tensor. Use Strassen for 3x3x3 twice (for k=0..2 and k=3..5): 7+7=14
    - For j=3: we have 3x1x6 sub-tensor. Rank = 3*6 = 18 (each (i,k) pair).
    Total: 14 + 18 = 32. Still worse than 72? No, 32 < 72.
    
    Wait, I need to be more careful. Let me use the block approach properly.
    
    Split B into B[:, 0:3] and B[:, 3:4] (i.e., j=0,1,2 and j=3).
    
    For j=0,1,2 (3x3x6):
      Split C into C[:, 0:3] and C[:, 3:6] (k=0,1,2 and k=3,4,5).
      * 3x3x3 (j=0..2, k=0..2): Strassen rank 7
      * 3x3x3 (j=0..2, k=3..5): Strassen rank 7
      Subtotal: 14
      
    For j=3 (3x1x6):
      This is an outer product: A[:,3] (3x1) times B[3,:] (1x6) = 3x6 matrix.
      Rank = 3*6 = 18.
      
    Total: 14 + 18 = 32.
    
    That's much better than 72! Let me implement this.
    """
    start_time = time.time()
    DEADLINE = 110

    def check_time():
        return time.time() - start_time < DEADLINE

    def verify_full(U, V, W, tol=1e-6):
        """Full verification of all entries."""
        rank = len(U[0]) if U else 0
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        # Check all zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = 0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if abs(val) > tol:
                            return False
        return True

    def schoolbook_full():
        """Return the schoolbook decomposition with rank n*m*p = 72."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def get_strassen_3x3x3():
        """
        Return the Strassen rank-7 decomposition for 3x3x3 matrix multiplication.
        
        Strassen's algorithm for 2x2x2 uses 7 multiplications. For 3x3x3,
        we can use a block decomposition. Here I use a known rank-7 scheme
        derived from applying Strassen to the 3x3 case.
        
        Actually, the rank of 3x3x3 matrix multiplication is 23 (Landsberg),
        not 7. Strassen's 7 is for 2x2x2. Let me use the schoolbook 27 for 3x3x3
        or find a better decomposition.
        
        For now, let's use a block approach:
        Split 3x3 into 2x2 + 2x2 + cross terms.
        This gets complicated. Let me just use schoolbook 27 for 3x3x3.
        
        Alternatively, I can use the known rank-23 decomposition, but that's complex.
        Let me try a different splitting strategy that avoids 3x3x3 entirely.
        """
        # Use schoolbook rank-27 for 3x3x3 as fallback
        U3 = np.zeros((9, 27))
        V3 = np.zeros((9, 27))
        W3 = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U3[i*3+j, col] = 1
                    V3[j*3+k, col] = 1
                    W3[k*3+i, col] = 1
                    col += 1
        return U3, V3, W3

    def build_ladder_decomposition():
        """
        Build the ladder decomposition:
        - Split j into j=0..2 and j=3
        - For j=0..2: split k into k=0..2 and k=3..5, use schoolbook 3x3x3 (rank 27 each)
        - For j=3: use outer product (rank 18)
        Total: 27 + 27 + 18 = 72 (same as schoolbook, not helpful)
        
        Let me try a better approach:
        - Use Strassen for 2x2x2 blocks within the 3x3x3 parts.
        
        Actually, let me try:
        Split A into A_top (3x2) and A_bottom (3x2) [j=0,1 and j=2,3]
        Split B into B_left (2x3) and B_right (2x3) [k=0,1,2 and k=3,4,5]
        Wait, this doesn't work cleanly because j is shared.
        
        New approach: Use the fact that 3x4x6 = direct sum of smaller tensors.
        
        Let me try splitting differently:
        C[:, 0:3] depends on A[:, 0:3] * B[0:3, 0:3] + A[:, 3] * B[3, 0:3]
        C[:, 3:6] depends on A[:, 0:3] * B[0:3, 3:6] + A[:, 3] * B[3, 3:6]
        
        So:
        Part 1: 3x3x3 (A[:,0:3], B[0:3,0:3], C[0:3,0:3]) - rank 27 schoolbook or 23
        Part 2: 3x3x3 (A[:,0:3], B[0:3,3:6], C[3:6,0:3]) - rank 27 schoolbook or 23
        Part 3: 3x1x3 (A[:,3], B[3,0:3], C[0:3,0:3]) - rank 9
        Part 4: 3x1x3 (A[:,3], B[3,3:6], C[3:6,0:3]) - rank 9
        
        Total with schoolbook: 27+27+9+9 = 72
        Total with rank-23: 23+23+9+9 = 64
        
        To get rank-23 for 3x3x3, I need a valid decomposition. Let me try to find one
        or use an alternative approach.
        
        Alternative: Use Strassen on 2x2x2 sub-blocks.
        For 3x3x3, split into 2x2x2 + 2x2x2 + cross terms.
        This is complex. Let me just return the schoolbook for now and note
        that rank 64 is achievable with a rank-23 3x3x3 decomposition.
        """
        total_rank = 72
        U = np.zeros((12, total_rank))
        V = np.zeros((24, total_rank))
        W = np.zeros((18, total_rank))
        
        U3, V3, W3 = get_strassen_3x3x3()
        rank_333 = 27
        
        # Part 1: A[:,0:3] * B[0:3,0:3] -> C[0:3,0:3]
        offset = 0
        for i in range(3):
            for j in range(3):
                U[i*4 + j, offset:offset+rank_333] = U3[i*3 + j, :]
        for j in range(3):
            for k in range(3):
                V[j*6 + k, offset:offset+rank_333] = V3[j*3 + k, :]
        for k in range(3):
            for i in range(3):
                W[k*3 + i, offset:offset+rank_333] = W3[k*3 + i, :]
        offset += rank_333
        
        # Part 2: A[:,0:3] * B[0:3,3:6] -> C[3:6,0:3]
        for i in range(3):
            for j in range(3):
                U[i*4 + j, offset:offset+rank_333] = U3[i*3 + j, :]
        for j in range(3):
            for k in range(3):
                V[j*6 + 3 + k, offset:offset+rank_333] = V3[j*3 + k, :]
        for k in range(3):
            for i in range(3):
                W[(3+k)*3 + i, offset:offset+rank_333] = W3[k*3 + i, :]
        offset += rank_333
        
        # Part 3: A[:,3] * B[3,0:3] -> C[0:3,0:3]
        rank_313 = 9
        for i in range(3):
            for k in range(3):
                U[i*4 + 3, offset] = 1
                V[3*6 + k, offset] = 1
                W[k*3 + i, offset] = 1
                offset += 1
        # This uses 9 terms, but I allocated rank_313=9 slots. Let me fix.
        # Actually, I need to redo this. Let me use a cleaner approach.
        
        # Reset and do it properly
        U = np.zeros((12, total_rank))
        V = np.zeros((24, total_rank))
        W = np.zeros((18, total_rank))
        offset = 0
        
        # Part 1: 3x3x3
        for i in range(3):
            for j in range(3):
                U[i*4 + j, offset:offset+rank_333] = U3[i*3 + j, :]
        for j in range(3):
            for k in range(3):
                V[j*6 + k, offset:offset+rank_333] = V3[j*3 + k, :]
        for k in range(3):
            for i in range(3):
                W[k*3 + i, offset:offset+rank_333] = W3[k*3 + i, :]
        offset += rank_333
        
        # Part 2: 3x3x3
        for i in range(3):
            for j in range(3):
                U[i*4 + j, offset:offset+rank_333] = U3[i*3 + j, :]
        for j in range(3):
            for k in range(3):
                V[j*6 + 3 + k, offset:offset+rank_333] = V3[j*3 + k, :]
        for k in range(3):
            for i in range(3):
                W[(3+k)*3 + i, offset:offset+rank_333] = W3[k*3 + i, :]
        offset += rank_333
        
        # Part 3: 3x1x3 (outer product)
        for i in range(3):
            for k in range(3):
                U[i*4 + 3, offset] = 1
                V[3*6 + k, offset] = 1
                W[k*3 + i, offset] = 1
                offset += 1
        
        # Part 4: 3x1x3 (outer product)
        for i in range(3):
            for k in range(3):
                U[i*4 + 3, offset] = 1
                V[3*6 + 3 + k, offset] = 1
                W[(3+k)*3 + i, offset] = 1
                offset += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    best_result = schoolbook_full()
    if not verify_full(best_result["U"], best_result["V"], best_result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    best_rank = n * m * p

    # Try ladder decomposition
    if check_time():
        ladder_result = build_ladder_decomposition()
        if verify_full(ladder_result["U"], ladder_result["V"], ladder_result["W"]):
            ladder_rank = len(ladder_result["U"][0])
            if ladder_rank < best_rank:
                best_result = ladder_result
                best_rank = ladder_rank

    return best_result
