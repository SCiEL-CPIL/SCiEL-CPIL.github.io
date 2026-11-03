import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s
    
    # Strategy: Use the rank-19 decomposition for 5x5x5 matmul tensor
    # Then extend to 3x5x5 by treating each output row separately but
    # sharing the 5x5 structure.
    #
    # For 5x5x5: T5[a,b,c] = 1 for a,b,c in 0..4
    # Rank-19 decomposition: U5 (25x19), V5 (25x19), W5 (25x19)
    # such that sum_r U5[a*5+b,r]*V5[b*5+c,r]*W5[c*5+a,r] = 1
    #
    # For 3x5x5: T3[i,j,k] = 1 for i in 0..2, j in 0..4, k in 0..4
    # We can use: for each i, the (j,k) part is the 5x5 matmul tensor
    # So if we have rank-19 for 5x5, we get rank 3*19 = 57 for 3x5x5
    # But we can potentially do better by sharing.
    #
    # Let me first construct the rank-19 decomposition for 5x5.
    # The known construction uses 4 Strassen 2x2x2 blocks + corrections.
    #
    # Actually, let me use a direct approach: construct the rank-19
    # decomposition using a known method.
    
    # Build rank-19 decomposition for 5x5x5
    # Use overlapping Strassen blocks and solve for corrections
    #
    # Method: Start with schoolbook (rank 25), then apply Strassen
    # reductions. 5x5 can fit:
    # - 4 non-overlapping 2x2x2 blocks: saves 4, rank 21
    # - Plus additional overlapping Strassen-like reductions
    #
    # Known result: rank 19 for 5x5x5 (Burgin 1982, independently by others)
    #
    # Let me use a construction: 
    # Place Strassen blocks at (0,0,0), (0,2,0), (2,0,0), (2,2,0) for the
    # 4x4 subcube, plus handle the 5th row/column separately.
    # 4x4x4 with 4 Strassen blocks: rank 4*7 + (16-4*8+4*7) ... 
    # 4x4x4: 64 terms, 4 Strassen blocks cover 4*8=32 terms with rank 28
    # Remaining 32 terms: rank 32. Total: 60. That's worse than 25!
    #
    # I'm confusing myself. Let me be very careful.
    #
    # For 5x5x5 matmul tensor: T[a,b,c] = 1 for all a,b,c in {0,1,2,3,4}
    # Schoolbook: 25 rank-1 terms, one for each (a,c) pair with b summed
    # Wait no. The matmul tensor is T[a,b,c] where C[a,c] = sum_b A[a,b]*B[b,c]
    # So T[a,b,c] = 1 for ALL (a,b,c). There are 5*5*5 = 125 entries, all = 1.
    # Schoolbook decomposition: for each (a,b,c), one rank-1 term. Rank = 125.
    # Strassen 2x2x2: replaces 8 schoolbook terms with 7, saving 1.
    #
    # For 5x5x5: how many non-overlapping 2x2x2 blocks?
    # floor(5/2)^3 = 2*2*2 = 8 blocks. Saves 8. Rank = 125-8 = 117.
    # That's still huge. The rank-19 result must use a much smarter construction.
    #
    # I think I've been misunderstanding. Let me re-read the problem.
    #
    # "Matrix multiplication 3,5,5" - multiply 3x5 by 5x5 to get 3x5.
    # The tensor is T[i,j,k] where i indexes rows of A (0..2), j indexes 
    # columns of A / rows of B (0..4), k indexes columns of B (0..4).
    # T[i,j,k] = 1 for all valid (i,j,k). Total: 3*5*5 = 75 entries.
    #
    # Schoolbook: rank 75.
    # Strassen blocks: 2x2x2 blocks. In 3x5x5:
    # i: floor(3/2) = 1 (i0=0 or i0=1)
    # j: floor(5/2) = 2 (j0=0,2)
    # k: floor(5/2) = 2 (k0=0,2)
    # Non-overlapping: 1*2*2 = 4 blocks. Saves 4. Rank = 71.
    #
    # The rank-19 for 5x5x5 is for the FULL 5x5x5 tensor (125 entries).
    # That's a different problem. For 3x5x5 (75 entries), the known bounds
    # are different.
    #
    # Let me just stick with the 4-Strassen-block approach (rank 71) and
    # try to improve by finding additional structure.
    #
    # Key insight from the strategy: model as linear map and use sparse
    # rank-reduction. Let me try to find if any schoolbook terms can be
    # "absorbed" into the Strassen terms.
    
    # Use 4 disjoint 2x2x2 Strassen blocks
    blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
    ]
    
    total_terms = n * m * p  # 75
    strassen_terms_count = len(blocks) * 8  # 32
    schoolbook_terms_count = total_terms - strassen_terms_count  # 43
    total_rank = len(blocks) * 7 + schoolbook_terms_count  # 28 + 43 = 71
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        """Add Strassen rank-7 decomposition for a 2x2x2 block."""
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Now try to improve: look for additional rank reductions
    # by finding linear dependencies among the current rank-1 terms.
    # 
    # Idea: if some subset of schoolbook terms can be expressed as a
    # combination of fewer rank-1 terms, we can reduce rank.
    #
    # For the 43 schoolbook terms, check if any 2 can be merged into 1.
    # Two rank-1 terms (u1,v1,w1) and (u2,v2,w2) can be replaced by one
    # rank-1 term iff the sum tensor is rank 1, which requires proportional
    # supports. In schoolbook, each term has a single non-zero entry in each
    # of U, V, W, so two terms can only merge if they share the same (i,j,k)
    # which is impossible.
    #
    # So simple pair merging won't work. Let me try a different approach:
    # use the remaining time to attempt a local search for better decompositions.
    
    if time.time() - start_time < DEADLINE - 10:
        # Try to find if we can add overlapping Strassen blocks in the i-dimension
        # Block at i0=1: covers i={1,2}, j={j0,j0+1}, k={k0,k0+1}
        # This overlaps with existing blocks at i0=0 on i=1.
        #
        # The combined decomposition would need to handle the overlap carefully.
        # Let me try: use blocks at BOTH i0=0 and i0=1, and solve for a
        # consistent decomposition.
        #
        # Actually, this is getting complex. Let me try a simpler improvement:
        # check if the current rank-71 decomposition is correct, and if so,
        # try to find a rank-70 or lower decomposition via perturbation.
        
        # Verify current decomposition
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(total_rank))
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-10:
            # Fall back to pure schoolbook
            total_rank_fb = n * m * p
            U_fb = [[0.0] * total_rank_fb for _ in range(n * m)]
            V_fb = [[0.0] * total_rank_fb for _ in range(m * p)]
            W_fb = [[0.0] * total_rank_fb for _ in range(p * n)]
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j][col_fb] = 1.0
                        V_fb[j*p+k][col_fb] = 1.0
                        W_fb[k*n+i][col_fb] = 1.0
                        col_fb += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
