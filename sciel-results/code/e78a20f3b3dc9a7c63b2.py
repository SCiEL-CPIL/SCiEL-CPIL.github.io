"""Combined strategy: Use Strassen decomposition for 2x2 blocks and schoolbook for the rest."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        from fractions import Fraction
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        # Target tensor T[i, a, k] = 1 for all i, a, k
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    # Vectors for U (i,a), V (a,k), W (k,i)
    # Indices: U: 00,01,10,11 -> 0,1,2,3
    #          V: 00,01,10,11 -> 0,1,2,3
    #          W: 00,01,10,11 -> 0,1,2,3 (k*n+i, n=2)
    
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    # Verify Strassen decomposition internally
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i*2 + a]
                        v_val = S_V[r][a*2 + k]
                        w_val = S_W[r][k*2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build candidate with rank 32
    # Strategy: Split B into 4 blocks of 2 columns.
    # For each block (cols k0, k1):
    #   - Use Strassen for A[:, 0:2] * B[0:2, k0:k1+1] (rank 7)
    #   - Use outer product for A[:, 2] * B[2, k0:k1+1] (rank 1)
    # Total rank = 4 * (7 + 1) = 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for block_idx in range(4):
        k0 = block_idx * 2
        k1 = k0 + 1
        
        # Part 1: Strassen for A[:, 0:2] * B[0:2, k0:k1+1]
        for r in range(7):
            # U indices: i in 0,1; a in 0,1 (global a = local a)
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val == 0: continue
                    u_idx = li * m + la
                    U[u_idx][col] = u_val
            
            # V indices: a in 0,1 (global a = local a); k in k0, k1 (global k = k0 + lk)
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val == 0: continue
                    v_idx = la * p + (k0 + lk)
                    V[v_idx][col] = v_val
            
            # W indices: k in k0, k1 (global k = k0 + lk); i in 0,1 (global i = li)
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val == 0: continue
                    w_idx = (k0 + lk) * n + li
                    W[w_idx][col] = w_val
            
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
        
        # Part 2: Outer product for A[:, 2] * B[2, k0:k1+1]
        # A[:, 2] is 2x1, B[2, k0:k1+1] is 1x2
        # This contributes to C[:, k0:k1+1]
        for li in range(2):
            for lk in range(2):
                u_idx = li * m + 2
                v_idx = 2 * p + (k0 + lk)
                w_idx = (k0 + lk) * n + li
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                # Wait, this is not a single rank-1 term.
                # The term A[:, 2] * B[2, k0:k1+1] is:
                # C[i, k] += A[i, 2] * B[2, k]
                # This can be decomposed into 2 rank-1 terms:
                #   term 1: A[i, 2] * B[2, k0] -> contributes to C[:, k0]
                #   term 2: A[i, 2] * B[2, k1] -> contributes to C[:, k1]
                # So we need 2 rank-1 terms, not 1.
                # Let's fix this.
                pass
        
        # Actually, let's redo Part 2 correctly.
        # We need 2 rank-1 terms for the outer product part.
        # But we only have 1 column slot left in this block (col is at 7, next is 8).
        # Total rank per block should be 8 (7 Strassen + 1 outer).
        # But the outer product A[:, 2] * B[2, k0:k1+1] is 2x2 matrix, which has rank 2.
        # So we need 2 rank-1 terms.
        # This means total rank per block is 7 + 2 = 9.
        # Total rank = 4 * 9 = 36.
        
        # Let's adjust the rank to 36.
        if col == 7:
            # We need to expand the matrices to rank 36
            new_rank = 36
            U = [row + [0] * (new_rank - rank) for row in U]
            V = [row + [0] * (new_rank - rank) for row in V]
            W = [row + [0] * (new_rank - rank) for row in W]
            rank = new_rank
        
        # Now add the 2 rank-1 terms for the outer product
        for lk in range(2):
            k_global = k0 + lk
            for li in range(2):
                u_idx = li * m + 2
                v_idx = 2 * p + k_global
                w_idx = k_global * n + li
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
        
        # Wait, this is still not right. The rank-1 term should be:
        # U[i, 2] * V[2, k] * W[k, i] = A[i, 2] * B[2, k] * 1
        # So for each k, we have a rank-1 term that contributes to C[:, k].
        # The rank-1 term is:
        #   U[i, 2] = A[i, 2] for i=0,1
        #   V[2, k] = B[2, k]
        #   W[k, i] = 1 for i=0,1
        # But we are constructing the decomposition of the tensor, not the actual matrix multiplication.
        # The tensor T[i, a, k] = 1 for all i, a, k.
        # So the rank-1 term for the outer product part is:
        #   U[i, 2] = 1 for i=0,1
        #   V[2, k] = 1
        #   W[k, i] = 1 for i=0,1
        # This is a valid rank-1 term.
        # So for each k, we have one rank-1 term.
        # So for 2 k's, we have 2 rank-1 terms.
        # Total rank per block = 7 + 2 = 9.
        # Total rank = 4 * 9 = 36.
        
        # Let's verify the rank is now 36.
        if block_idx == 0:
            rank = col  # Should be 9
            # But we need to ensure the matrices have the correct size.
            # The matrices were initialized with rank 32, then expanded to 36.
            # But col is now 9, so we have used 9 columns.
            # The remaining columns (9 to 35) are zero.
            # This is fine, but the rank is effectively 9.
            # However, the evaluator expects the rank to be the number of columns.
            # So we should trim the matrices to the actual rank.
    
    # Trim the matrices to the actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    candidate = {"U": U, "V": V, "W": W}
    
    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
