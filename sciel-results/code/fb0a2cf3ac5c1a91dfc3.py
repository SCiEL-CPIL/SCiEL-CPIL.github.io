import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s safety margin
    
    # We'll build the decomposition by tiling with 2x2x2 Strassen blocks
    # and handling uncovered (i,j,k) triples with schoolbook terms.
    # 
    # Strategy: For each (i,j,k) triple, we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # We'll use a greedy approach: try to place Strassen blocks that cover
    # 8 (i,j,k) triples each with rank 7, then fill gaps with schoolbook.
    #
    # Actually, let's think more carefully. The tensor T has entries T[i,j,k] = 1 for all i,j,k.
    # We want rank-minimal decomposition.
    #
    # Known results: For 3x5x7, the best known rank is around 90-100.
    # Schoolbook gives 105.
    # 
    # Let's try a structured approach:
    # - Use Strassen for overlapping 2x2x2 blocks
    # - The key is that Strassen blocks overlap in their contributions
    
    # Let's try: partition into non-overlapping 2x2x2 blocks where possible,
    # then handle the remainder.
    # 
    # n=3, m=5, p=7
    # We can fit: floor(3/2)=1 in i, floor(5/2)=2 in j, floor(7/2)=3 in k
    # So 1*2*3 = 6 blocks of size 2x2x2, using 12 i-rows, 8 j-cols, 6 k-cols... wait
    # Actually i has 3 rows, so one 2x2 block covers i=0,1, leaving i=2
    # j has 5 cols, two 2x2 blocks cover j=0,1 and j=2,3, leaving j=4
    # k has 7 cols, three 2x2 blocks cover k=0,1; k=2,3; k=4,5, leaving k=6
    #
    # This is getting complex. Let me just implement a clean schoolbook with
    # a few Strassen substitutions to get below 105.
    
    # Actually, let me implement a proper search-based approach.
    # Start with schoolbook rank 105, then try to merge terms.
    
    # For now, let's implement a hybrid: use Strassen for the 2x2x2 corner
    # blocks and schoolbook for the rest.
    
    # Let me just do a careful implementation:
    # Track which (i,j,k) triples are covered and by which rank terms.
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec, v_vec, w_vec are dicts mapping index->value."""
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # We'll represent vectors as dicts for sparsity
    # U indices: (i,j) -> i*m + j, range [0, n*m)
    # V indices: (j,k) -> j*p + k, range [0, m*p)
    # W indices: (k,i) -> k*n + i, range [0, p*n)
    
    # First, let's try a block-based approach.
    # For each combination of (i_block, j_block, k_block) where each is a 2-element
    # subset of the respective range, apply Strassen.
    
    # But blocks must be disjoint in their (i,j,k) coverage to avoid double-counting.
    # Actually, they CAN overlap if we carefully account for contributions.
    # This is the hard part.
    
    # Simpler approach: use non-overlapping blocks + schoolbook for the rest.
    # 
    # i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5,6}
    # 
    # Let's use blocks:
    # Block (i:0-1, j:0-1, k:0-1): Strassen, covers 8 triples, rank 7
    # Block (i:0-1, j:2-3, k:2-3): Strassen, covers 8 triples, rank 7  
    # Block (i:0-1, j:0-1, k:2-3): Strassen, covers 8 triples, rank 7
    # Block (i:0-1, j:2-3, k:4-5): Strassen, covers 8 triples, rank 7
    # Block (i:0-1, j:0-1, k:4-5): Strassen, covers 8 triples, rank 7
    # Block (i:0-1, j:4, k:0-1): schoolbook 1x2x2, rank 2
    # Block (i:0-1, j:4, k:2-3): schoolbook 1x2x2, rank 2
    # Block (i:0-1, j:4, k:4-5): schoolbook 1x2x2, rank 2
    # Block (i:0-1, j:4, k:6): schoolbook 1x1x1, rank 1
    # Block (i:2, j:0-1, k:0-1): schoolbook 1x2x2, rank 2
    # Block (i:2, j:2-3, k:2-3): schoolbook 1x2x2, rank 2
    # Block (i:2, j:0-1, k:2-3): schoolbook 1x2x2, rank 2
    # Block (i:2, j:2-3, k:4-5): schoolbook 1x2x2, rank 2
    # Block (i:2, j:0-1, k:4-5): schoolbook 1x2x2, rank 2
    # Block (i:2, j:4, k:0-1): schoolbook 1x1x2, rank 2
    # Block (i:2, j:4, k:2-3): schoolbook 1x1x2, rank 2
    # Block (i:2, j:4, k:4-5): schoolbook 1x1x2, rank 2
    # Block (i:2, j:4, k:6): schoolbook 1x1x1, rank 1
    # Block (i:0-1, j:2-3, k:6): schoolbook 2x2x1, rank 4
    # Block (i:0-1, j:0-1, k:6): schoolbook 2x2x1, rank 4
    # Block (i:2, j:0-1, k:6): schoolbook 1x2x1, rank 2
    # Block (i:2, j:2-3, k:6): schoolbook 1x2x1, rank 2
    
    # Total: 5*7 + 2+2+2+1+2+2+2+2+2+2+2+2+1+4+4+2+2 = 35+48 = 83? Let me recount.
    # Strassen blocks: 5 blocks * 7 = 35
    # i:0-1, j:4, k:0-1: 2
    # i:0-1, j:4, k:2-3: 2
    # i:0-1, j:4, k:4-5: 2
    # i:0-1, j:4, k:6: 1
    # i:2, j:0-1, k:0-1: 2
    # i:2, j:2-3, k:2-3: 2
    # i:2, j:0-1, k:2-3: 2
    # i:2, j:2-3, k:4-5: 2
    # i:2, j:0-1, k:4-5: 2
    # i:2, j:4, k:0-1: 2
    # i:2, j:4, k:2-3: 2
    # i:2, j:4, k:4-5: 2
    # i:2, j:4, k:6: 1
    # i:0-1, j:2-3, k:6: 4
    # i:0-1, j:0-1, k:6: 4
    # i:2, j:0-1, k:6: 2
    # i:2, j:2-3, k:6: 2
    # Sum: 35 + 2+2+2+1+2+2+2+2+2+2+2+2+1+4+4+2+2 = 35 + 40 = 75
    
    # That's already better than 105! Let me implement this.
    
    def add_strassen_block(i0, j0, k0):
        """Add Strassen decomposition for 2x2x2 block starting at (i0, j0, k0)."""
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # Global: i = i0+li, j = j0+lj, k = k0+lk
        
        # Strassen intermediate products:
        # P1 = (a00+a11)(b00+b11)
        # P2 = (a10+a11)b00
        # P3 = a00(b10-b11)
        # P4 = a11(b01-b00)
        # P5 = (a00+a01)b11
        # P6 = (a10-a00)(b00+b10)
        # P7 = (a01-a11)(b01+b11)
        
        # c00 = p1 - p4 + p5 + p7
        # c01 = p3 + p4
        # c10 = p2 + p5
        # c11 = p1 - p3 + p6 + p7
        
        # For each Strassen term, we need U, V, W vectors.
        # U is over (i,j) pairs, V over (j,k) pairs, W over (k,i) pairs.
        
        # Term 1: P1 = (a00+a11)(b00+b11)
        # U: {(i0,j0):1, (i0+1,j0+1):1}
        # V: {(j0,k0):1, (j0+1,k0+1):1}
        # W: {(k0,i0):1, (k0+1,i0+1):1}  -- contributes to c00 and c11
        u1 = {(i0, j0): 1, (i0+1, j0+1): 1}
        v1 = {(j0, k0): 1, (j0+1, k0+1): 1}
        w1 = {(k0, i0): 1, (k0+1, i0+1): 1}
        add_term(u1, v1, w1)
        
        # Term 2: P2 = (a10+a11)b00
        # U: {(i0+1,j0):1, (i0+1,j0+1):1}
        # V: {(j0,k0):1}
        # W: {(k0+1,i0):1}  -- contributes to c10
        u2 = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        v2 = {(j0, k0): 1}
        w2 = {(k0+1, i0): 1}
        add_term(u2, v2, w2)
        
        # Term 3: P3 = a00(b10-b11)
        # U: {(i0,j0):1}
        # V: {(j0+1,k0):1, (j0+1,k0+1):-1}
        # W: {(k0,i0+1):1}  -- contributes to c01
        u3 = {(i0, j0): 1}
        v3 = {(j0+1, k0): 1, (j0+1, k0+1): -1}
        w3 = {(k0, i0+1): 1}
        add_term(u3, v3, w3)
        
        # Term 4: P4 = a11(b01-b00)
        # U: {(i0+1,j0+1):1}
        # V: {(j0,k0+1):1, (j0,k0):-1}
        # W: {(k0,i0):1}  -- contributes to c00 (with -1)
        u4 = {(i0+1, j0+1): 1}
        v4 = {(j0, k0+1): 1, (j0, k0): -1}
        w4 = {(k0, i0): 1}
        add_term(u4, v4, w4)
        
        # Term 5: P5 = (a00+a01)b11
        # U: {(i0,j0):1, (i0,j0+1):1}
        # V: {(j0+1,k0+1):1}
        # W: {(k0,i0):1, (k0+1,i0):1}  -- contributes to c00 and c10
        u5 = {(i0, j0): 1, (i0, j0+1): 1}
        v5 = {(j0+1, k0+1): 1}
        w5 = {(k0, i0): 1, (k0+1, i0): 1}
        add_term(u5, v5, w5)
        
        # Term 6: P6 = (a10-a00)(b00+b10)
        # U: {(i0+1,j0):1, (i0,j0):-1}
        # V: {(j0,k0):1, (j0+1,k0):1}
        # W: {(k0+1,i0+1):1}  -- contributes to c11
        u6 = {(i0+1, j0): 1, (i0, j0): -1}
        v6 = {(j0, k0): 1, (j0+1, k0): 1}
        w6 = {(k0+1, i0+1): 1}
        add_term(u6, v6, w6)
        
        # Term 7: P7 = (a01-a11)(b01+b11)
        # U: {(i0,j0+1):1, (i0+1,j0+1):-1}
        # V: {(j0,k0+1):1, (j0+1,k0+1):1}
        # W: {(k0,i0):1, (k0+1,i0+1):1}  -- contributes to c00 and c11
        u7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        v7 = {(j0, k0+1): 1, (j0+1, k0+1): 1}
        w7 = {(k0, i0): 1, (k0+1, i0+1): 1}
        add_term(u7, v7, w7)
    
    def add_schoolbook_block(i_list, j_list, k_list):
        """Add schoolbook terms for given index lists."""
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    add_term(u, v, w)
    
    # Define blocks
    # Strassen blocks (2x2x2):
    strassen_blocks = [
        (0, 0, 0),   # i:0-1, j:0-1, k:0-1
        (0, 2, 2),   # i:0-1, j:2-3, k:2-3
        (0, 0, 2),   # i:0-1, j:0-1, k:2-3
        (0, 2, 4),   # i:0-1, j:2-3, k:4-5
        (0, 0, 4),   # i:0-1, j:0-1, k:4-5
    ]
    
    # Schoolbook blocks:
    schoolbook_blocks = [
        # (i_list, j_list, k_list)
        ([0,1], [4], [0,1]),    # i:0-1, j:4, k:0-1
        ([0,1], [4], [2,3]),    # i:0-1, j:4, k:2-3
        ([0,1], [4], [4,5]),    # i:0-1, j:4, k:4-5
        ([0,1], [4], [6]),      # i:0-1, j:4, k:6
        ([2], [0,1], [0,1]),    # i:2, j:0-1, k:0-1
        ([2], [2,3], [2,3]),    # i:2, j:2-3, k:2-3
        ([2], [0,1], [2,3]),    # i:2, j:0-1, k:2-3
        ([2], [2,3], [4,5]),    # i:2, j:2-3, k:4-5
        ([2], [0,1], [4,5]),    # i:2, j:0-1, k:4-5
        ([2], [4], [0,1]),      # i:2, j:4, k:0-1
        ([2], [4], [2,3]),      # i:2, j:4, k:2-3
        ([2], [4], [4,5]),      # i:2, j:4, k:4-5
        ([2], [4], [6]),        # i:2, j:4, k:6
        ([0,1], [2,3], [6]),    # i:0-1, j:2-3, k:6
        ([0,1], [0,1], [6]),    # i:0-1, j:0-1, k:6
        ([2], [0,1], [6]),      # i:2, j:0-1, k:6
        ([2], [2,3], [6]),      # i:2, j:2-3, k:6
    ]
    
    # Add all blocks
    for ib, jb, kb in strassen_blocks:
        add_strassen_block(ib, jb, kb)
    
    for il, jl, kl in schoolbook_blocks:
        add_schoolbook_block(il, jl, kl)
    
    # Convert from dict format to list format
    nU = n * m
    nV = m * p
    nW = p * n
    rank = len(U)
    
    U_mat = [[0] * rank for _ in range(nU)]
    V_mat = [[0] * rank for _ in range(nV)]
    W_mat = [[0] * rank for _ in range(nW)]
    
    for r in range(rank):
        for (i, j), val in U[r].items():
            U_mat[i * m + j][r] = val
        for (j, k), val in V[r].items():
            V_mat[j * p + k][r] = val
        for (k, i), val in W[r].items():
            W_mat[k * n + i][r] = val
    
    # Verify correctness
    # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    U_arr = np.array(U_mat, dtype=float)
    V_arr = np.array(V_mat, dtype=float)
    W_arr = np.array(W_mat, dtype=float)
    
    # For each (i,j,k), compute the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    # Fallback to schoolbook if verification fails
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj][col] = 1
                                V_fb[jj * p + kk][col] = 1
                                W_fb[kk * n + ii][col] = 1
                                col += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
