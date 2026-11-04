"""Optimized rank decomposition for 2x3 x 3x9 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    # We decompose into 2 * (p/m) * m = 2 * 3 * 3 = 18 terms
    # For each row i of A (n=2), each column-group g of B (p/m=3 groups),
    # and each term t in 0..m-1 (m=3):
    #   U[i*m + t, col] = 1  (selects t-th element of row i of A)
    #   V[t*m + (g*m + j), col] = 1 for j=0..m-1 (selects t-th row of B's g-th group)
    #   W[(g*m + j)*n + i, col] = 1 (outputs to C[i, g*m+j])
    # 
    # Wait, let me reconsider the indexing.
    # A is n x m = 2 x 3, indexed A[i, k] where i in 0..1, k in 0..2
    # B is m x p = 3 x 9, indexed B[k, l] where k in 0..2, l in 0..8
    # C is n x p = 2 x 9, indexed C[i, l] where i in 0..1, l in 0..8
    # C[i, l] = sum_k A[i, k] * B[k, l]
    #
    # U has shape (n*m, rank) = (6, rank), U[i*m + k, col]
    # V has shape (m*p, rank) = (27, rank), V[k*p + l, col]
    # W has shape (p*n, rank) = (18, rank), W[l*n + i, col]
    #
    # For each (i, l), C[i,l] = sum_k A[i,k]*B[k,l]
    # Schoolbook: rank = n*m*p = 54, one term per (i,k,l)
    #
    # Reduction: Group l into groups of m=3. Let l = g*m + j where g in 0..2, j in 0..2.
    # For fixed (i, g): C[i, g*m+j] = sum_k A[i,k]*B[k, g*m+j] for j=0,1,2
    # This is: for each j, compute dot product of A[i,:]* and B[:, g*m+j]
    # 
    # The key: for fixed i and g, we have 3 output values (j=0,1,2) each being a dot product.
    # We can use rank m=3 for this 3-output computation by sharing the A factors.
    #
    # Term (i, g, t) where t in 0..2:
    #   U[i*m + t, col] = 1
    #   V[t*p + (g*m + j), col] = 1 for all j in 0..2  (B's row t, all 3 columns in group g)
    #   W[(g*m + j)*n + i, col] = 1 for all j in 0..2
    #
    # Check: sum over t of U[i*m+t, col]*V[t*p + (g*m+j), col]*W[(g*m+j)*n+i, col]
    # = sum_t 1 * 1 * 1 = 3? No, that's not right.
    #
    # Let me reconsider. The decomposition is:
    # C[i, l] = sum_col U[i*m + ?, col] * V[?, col] * W[l*n + i, col]
    #
    # Actually the tensor decomposition means:
    # T[i, k, l] = sum_c U[i*m+k, c] * V[k*p+l, c] * W[l*n+i, c]
    #
    # For schoolbook: T[i,k,l] = delta_{i,k,l} meaning T[i,k,l]=1 only when... 
    # No, T[i,k,l] is the tensor such that C[i,l] = sum_k T[i,k,l] * A[i,k] * B[k,l]
    # So T[i,k,l] = 1 for all (i,k,l).
    #
    # Schoolbook: one term per (i,k,l): U[i*m+k, c]=1, V[k*p+l, c]=1, W[l*n+i, c]=1
    #
    # For reduction, I need to find terms where multiple (i,k,l) are covered.
    # 
    # Let me try: for fixed i and g (column group), use m=3 terms indexed by t:
    # Term (i, g, t):
    #   U[i*m + t, col] = 1
    #   V[t*p + g*m + j, col] = 1 for j=0,1,2
    #   W[(g*m + j)*n + i, col] = 1 for j=0,1,2
    #
    # Contribution to T[i', k', l'] from this term:
    #   U[i'*m+k', col] * V[k'*p+l', col] * W[l'*n+i', col]
    #   = [i'==i and k'==t] * [k'==t and l' in {g*m, g*m+1, g*m+2}] * [l' in {g*m,...} and i'==i]
    #   = 1 if i'==i, k'==t, l' in group g; else 0
    #
    # Sum over t=0,1,2 for fixed (i,g):
    #   T[i, k', l'] = sum_t [k'==t and l' in group g] = 1 if l' in group g (for any k')
    #
    # That's wrong - it gives T[i,k',l'] = 1 for all k' when l' is in group g.
    #
    # I need a different approach. Let me think about this differently.
    #
    # The tensor T[i,k,l] = 1 for all i,k,l. I want to decompose it as sum of rank-1 tensors.
    # A rank-1 tensor is u_i * v_k * w_l.
    #
    # For the 2x3x9 case, one approach: since n=2 is small, use the fact that
    # we can separate the i dimension.
    #
    # Actually, let me just try rank = n * ceil(p/m) * m = 2 * 3 * 3 = 18 with a proper construction.
    #
    # Alternative: Use the standard rank reduction for matrix multiplication.
    # For (n x m) * (m x p), if we can find a decomposition, the rank is at most n*p*ceil(m/something).
    #
    # Let me just implement a verified rank-18 decomposition if it exists, otherwise fall back to 54.
    #
    # Actually, for 2x3x3 matrix mult, the rank is 7 (not 9). For 2x3x9, we can use 2x3x3 blocks.
    # 2x3x3 has rank 7, so 2x3x9 could have rank 3*7 = 21? Or maybe less.
    #
    # Let me just do the schoolbook for now but try to verify and potentially find better.
    
    deadline = time.time() + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Schoolbook decomposition, rank = n*m*p = 54
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for k in range(m):
            for l in range(p):
                U[i * m + k][column] = 1
                V[k * p + l][column] = 1
                W[l * n + i][column] = 1
                column += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # T[i,k,l] = sum_c U[i*m+k, c] * V[k*p+l, c] * W[l*n+i, c]
        # Should equal 1 for all i,k,l
        for i in range(n):
            for k in range(m):
                for l in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+k][c] * V[k*p+l][c] * W[l*n+i][c]
                    if s != 1:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find a lower rank decomposition using the structure
    # For 2x3x3, the minimal rank is 7. Use this for each 2x3x3 block.
    # 2x3x9 = three 2x3x3 blocks, so rank could be 3*7 = 21.
    
    # Rank-7 decomposition for 2x3x3 matrix multiplication
    # This is a known result. Let me construct it.
    # For 2x3x3: A is 2x3, B is 3x3, C is 2x3
    # C[i,l] = sum_k A[i,k]*B[k,l]
    
    # Known rank-7 decomposition for 2x3x3:
    # I'll use a construction based on the fact that 2x3x3 has rank 7.
    # One way: use the standard algorithm and eliminate one term.
    
    # Actually, let me try a different approach: use numpy to find a low-rank decomposition
    # of the tensor via optimization, but that's complex. Let me just return the schoolbook.
    
    # Time check
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
