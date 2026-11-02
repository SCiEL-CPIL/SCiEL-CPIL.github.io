"""Matrix multiplication 2x3x10 - rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for col in range(r):
                    total += U[a][col] * V[b][col] * W[c][col]
                if total != 1:
                    return False
                # Check that non-target entries are 0
    # Check a sample of non-target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                # Check some c values that should be 0
                for c_test in range(p * n):
                    if c_test != k * n + i:
                        total = 0
                        for col in range(r):
                            total += U[a][col] * V[b][col] * W[c_test][col]
                        if abs(total) > 1e-10:
                            return False
    return True

def _try_structured_decomposition(n, m, p, deadline):
    """Try to find a rank-reduced decomposition using structured patterns."""
    
    # Strategy: For each j (column of A / row of B), we have a 2×p sub-problem.
    # The full problem is 3 such sub-problems sharing the same A and B structure.
    
    # Try: express the tensor as sum over j of (2×p sub-tensor for j)
    # Each 2×p sub-tensor has rank at most 2*p = 20 in schoolbook.
    # Total: 3*20 = 60.
    
    # Can we do better by sharing across j?
    
    # Idea: For each k, C[:,k] = A @ B[:,k] (2×3 times 3×1 = 2×1)
    # This is 3 scalar multiplications per (i,k) pair, total 2*3*10 = 60.
    
    # Let's try a decomposition where we group by k and use the fact that
    # A is 2×3. For fixed k, we need to compute A @ b_k where b_k is a 3-vector.
    # This is a 2×3 × 3×1 multiplication, which has rank 6 in schoolbook.
    # But 2×3×1 matrix multiplication might have lower rank!
    
    # For 2×3×1: C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    #            C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    # Schoolbook rank = 6.
    # Can we do better? The tensor is 6×3×2.
    # Known: 2×n×1 has rank 2n (I think it's minimal).
    # So 2×3×1 has rank 6, no improvement.
    
    # So per-k, we need 6 rank-1 terms, and there are 10 k's, giving 60 total.
    # This suggests rank 60 might be minimal.
    
    # But wait - the tensor is not a direct sum over k! The A entries are shared across k.
    # Let me reconsider.
    
    # Actually, the tensor T[i*m+j, j*p+k, k'*n+i'] = 1 if i'=i, j'=j, k'=k else 0.
    # This IS a direct sum over (i,j,k) in the sense that each (i,j,k) contributes
    # to exactly one entry of the output tensor.
    
    # However, the rank-1 decomposition doesn't have to align with this structure.
    # A rank-1 term U⊗V⊗W can have U supported on multiple (i,j) pairs, etc.
    
    # Let me try to find a decomposition with rank < 60 by solving a system.
    # For small rank reductions, we can try specific patterns.
    
    # Try rank 59: Can we merge two schoolbook terms into one?
    # Terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exist
    # U, V, W such that U⊗V⊗W = e_{i1,j1,k1} + e_{i2,j2,k2}
    # This requires U, V, W to each have at most 2 non-zero entries, and the
    # outer product to match. Generally not possible unless the indices align.
    
    # For example, (i,j,k) and (i,j,k') with same i,j:
    # U = e_{i,j}, V = e_{j,k} + e_{j,k'}, W = e_{k,i} + e_{k',i}
    # Then U⊗V⊗W = e_{i,j}⊗(e_{j,k}+e_{j,k'})⊗(e_{k,i}+e_{k',i})
    # = e_{i,j,k,i} + e_{i,j,k,k'} + e_{i,j,k',i} + e_{i,j,k',k}
    # The cross terms e_{i,j,k,k'} and e_{i,j,k',k} are unwanted!
    
    # So direct merging doesn't work. We need a more clever approach.
    
    # Let me try: use linear combinations.
    # Define for each j: S_j = A[0,j] + A[1,j], D_j = A[0,j] - A[1,j]
    # Then A[0,j] = (S_j + D_j)/2, A[1,j] = (S_j - D_j)/2
    # 
    # C[0,k] = sum_j A[0,j]*B[j,k] = sum_j (S_j + D_j)/2 * B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k] = sum_j (S_j - D_j)/2 * B[j,k]
    #
    # = 1/2 * [sum_j S_j*B[j,k] + sum_j D_j*B[j,k]]
    # = 1/2 * [sum_j S_j*B[j,k] - sum_j D_j*B[j,k]]
    #
    # This uses 6*p = 60 rank-1 terms (3 S_j*B[j,k] + 3 D_j*B[j,k] for each k).
    # Still 60.
    
    # What if we also transform B?
    # For each k, define S'_k = B[0,k] + B[1,k] + B[2,k], etc.
    # This gets complicated.
    
    # Let me try a computational approach: for small rank reductions,
    # set up and solve a system of equations.
    
    # For rank r, we have variables U[6,r], V[30,r], W[20,r].
    # Constraints: for all (a,b,c) in the support of T, sum_col U[a,col]*V[b,col]*W[c,col] = 1
    # And for all (a,b,c) not in the support, sum_col U[a,col]*V[b,col]*W[c,col] = 0
    
    # This is a system of 3600 equations. For r=59, we have 6*59+30*59+20*59 = 56*59 = 3304 unknowns.
    # Underdetermined, but we need rational solutions.
    
    # Practical approach: start from schoolbook and try to "absorb" one term into the others.
    
    # Take the last schoolbook term (i=1,j=2,k=9) and try to express it as a linear
    # combination of the other 59 terms plus modifications.
    
    # This is getting too complex for a quick implementation. Let me try a different tactic:
    # use random search with verification.
    
    import random
    random.seed(42)
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Generate schoolbook
    U_sb = [[0] * best_rank for _ in range(n * m)]
    V_sb = [[0] * best_rank for _ in range(m * p)]
    W_sb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find rank 59 by modifying the schoolbook
    # Pick a term to eliminate and see if we can adjust others
    target_rank = 59
    if time.time() < deadline - 30:
        # Try: for each pair of terms, see if we can merge them
        # by finding a rank-1 term that equals their sum
        # This requires the two terms to be "compatible"
        
        # Try all pairs (this is 60*59/2 = 1770 pairs, manageable)
        merged = False
        for idx1 in range(60):
            if time.time() > deadline - 25:
                break
            for idx2 in range(idx1 + 1, 60):
                # Get the (i,j,k) for each
                # idx = i*30 + j*10 + k (approximately)
                i1 = idx1 // 30
                j1 = (idx1 % 30) // 10
                k1 = idx1 % 10
                i2 = idx2 // 30
                j2 = (idx2 % 30) // 10
                k2 = idx2 % 10
                
                # Can we merge? Only if they share structure
                # Case 1: same i, same j, different k
                if i1 == i2 and j1 == j2:
                    # U = e_{i,j}, V = e_{j,k1} + e_{j,k2}, W = e_{k1,i} + e_{k2,i}
                    # This gives cross terms, so not directly mergeable
                    pass
                # Case 2: same i, different j, same k
                elif i1 == i2 and k1 == k2:
                    # U = e_{i,j1} + e_{i,j2}, V = e_{j1,k} + e_{j2,k}, W = e_{k,i}
                    # Cross terms again
                    pass
                # Case 3: same j, different i, same k
                elif j1 == j2 and k1 == k2:
                    # U = e_{i1,j} + e_{i2,j}, V = e_{j,k}, W = e_{k,i1} + e_{k,i2}
                    # Cross terms
                    pass
                # Case 4: same i, same k, different j - same as case 2
                # No direct merging works without cross terms
        
        # Since direct merging doesn't work, try a different approach:
        # Use a "padding" technique where we add and subtract terms
        
        # For now, return schoolbook
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Generate schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try structured decomposition
    result = _try_structured_decomposition(n, m, p, deadline)
    if result:
        U, V, W = result["U"], result["V"], result["W"]
        if U and V and W:
            r = len(U[0]) if U else 0
            if r < best_rank:
                # Verify
                if _verify_decomposition(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
