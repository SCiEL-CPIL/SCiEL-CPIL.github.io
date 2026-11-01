"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # The multiplication tensor T for (n,m,p) = (2,3,8):
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    # We want T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # i.e., T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
    
    # Strategy: Try to find a low-rank decomposition
    # The tensor has size 6 x 24 x 16 = 2304 entries
    # Each rank-1 term has 6+24+16 = 46 parameters
    # So we need r*46 >= 2304 for a non-trivial solution, r >= 50
    # But the tensor is sparse and structured, so lower rank is possible
    
    # Let's try to construct a rank-40 solution using the structure
    # Key insight: for each j (0,1,2), the terms involving j are:
    #   For i in {0,1}, k in {0,...,7}: U[i,j,d] * V[j,k,d] * W[k,i,d]
    # This gives a rank-1 coupling for each (i,j,k) triple
    
    # Schoolbook: 48 terms, one for each (i,j,k)
    # We can share terms when the same d is used for multiple (i,j,k)
    
    # Let me try a different approach: use the known rank-11 for 2x3*3x2
    # and rank-16 for 2x3*3x3, then combine
    
    # Actually, let me try to directly construct a solution
    # using the structure of the tensor
    
    # For a 2x3 * 3x8 multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C is 2x8, A is 2x3, B is 3x8
    
    # Let's denote the entries:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01, ..., b07],
    #      [b10, b11, ..., b17],
    #      [b20, b21, ..., b27]]
    # C = [[c00, c01, ..., c07],
    #      [c10, c11, ..., c17]]
    
    # c[i,k] = a[i,0]*b[0,k] + a[i,1]*b[1,k] + a[i,2]*b[2,k]
    
    # Schoolbook uses 3*2*8 = 48 products
    # Can we do better?
    
    # One approach: group the b[j,k] terms
    # For fixed i, c[i,k] = sum_j a[i,j] * b[j,k]
    # This is a dot product of a[i,:] with B[:,k]
    
    # Let me try to use a bilinear algorithm
    # Define new variables to share computation
    
    # For the 2x3 * 3x2 subproblem (rank 11):
    # This is a known result. Let me try to construct it.
    
    # Actually, let me just use the schoolbook for now and verify it works,
    # then try to optimize.
    
    rank = n * m * p  # 48 schoolbook
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
    
    # Verify the decomposition
    # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
    # For a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T should be 1 iff j==j' and k==k' and i==i'
    
    # Let's verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                assert val == 1, f"Failed at ({i},{j},{k}): got {val}"
    
    # Now try to reduce the rank
    # Check if any terms can be combined
    
    # Let me try a rank-40 approach
    # Idea: for each pair (i,k), we need sum_j a[i,j]*b[j,k]
    # We can compute partial products and share them
    
    # Let me try a different construction
    # Use the fact that we only have 2 rows in A
    
    # For row 0: c[0,k] = a[0,0]*b[0,k] + a[0,1]*b[1,k] + a[0,2]*b[2,k]
    # For row 1: c[1,k] = a[1,0]*b[0,k] + a[1,1]*b[1,k] + a[1,2]*b[2,k]
    
    # Define shared products:
    # p[j,k] = b[j,k] (these are inputs, not products)
    # Actually, the bilinear algorithm uses products of A entries and B entries
    
    # Let me try:
    # t1 = a[0,0] * b[0,k] for all k -> 8 terms
    # t2 = a[0,1] * b[1,k] for all k -> 8 terms
    # t3 = a[0,2] * b[2,k] for all k -> 8 terms
    # t4 = a[1,0] * b[0,k] for all k -> 8 terms
    # t5 = a[1,1] * b[1,k] for all k -> 8 terms
    # t6 = a[1,2] * b[2,k] for all k -> 8 terms
    # Total: 48 terms (schoolbook)
    
    # To reduce, we need to share. For example:
    # (a[0,0]+a[1,0]) * b[0,k] gives us a[0,0]*b[0,k] + a[1,0]*b[0,k]
    # But we need them separately for c[0,k] and c[1,k]
    
    # We can use:
    # s1 = (a[0,0] + a[1,0]) * b[0,k]
    # s2 = a[0,0] * b[0,k]
    # Then a[1,0]*b[0,k] = s1 - s2
    # This saves 1 term per k for the b[0,k] column
    
    # Similarly for other j values.
    # For each j, we can save 1 term by using this trick.
    # Total savings: 3 terms (one for each j)
    # New rank: 48 - 3 = 45
    
    # Let me construct this rank-45 solution
    
    rank45 = 45
    U45 = [[0] * rank45 for _ in range(n * m)]
    V45 = [[0] * rank45 for _ in range(m * p)]
    W45 = [[0] * rank45 for _ in range(p * n)]
    
    col = 0
    # For each j and k, we have:
    # Term 1: a[0,j] * b[j,k] -> contributes to c[0,k]
    # Term 2: (a[0,j] + a[1,j]) * b[j,k] -> contributes to both c[0,k] and c[1,k]
    # Then a[1,j]*b[j,k] = term2 - term1 -> contributes to c[1,k]
    
    # So for each (j,k):
    # U[0*3+j][col] = 1, V[j*8+k][col] = 1, W[k*2+0][col] = 1
    # U[0*3+j][col+1] = 1, U[1*3+j][col+1] = 1, V[j*8+k][col+1] = 1, W[k*2+0][col+1] = 1, W[k*2+1][col+1] = 1
    # The difference gives a[1,j]*b[j,k] for c[1,k]
    
    # Wait, this doesn't quite work because the W vectors need to be consistent.
    
    # Let me think more carefully.
    # We want: c[i,k] = sum_j a[i,j] * b[j,k]
    # 
    # For fixed (j,k), define:
    #   x1 = a[0,j] * b[j,k]  -> W contribution: W[k*2+0] += 1
    #   x2 = (a[0,j] + a[1,j]) * b[j,k] -> W contribution: W[k*2+0] += 1, W[k*2+1] += 1
    #   Then a[1,j]*b[j,k] = x2 - x1 -> W contribution: W[k*2+1] += 1, W[k*2+0] -= 1
    #
    # But we can't have negative W entries easily in this framework... actually we can, entries can be any integer.
    
    # Let me try:
    # For each (j,k), use 2 terms instead of 2 terms (one for i=0, one for i=1):
    # Term A: U[0,j] = 1, V[j,k] = 1, W[k,0] = 1  (gives a[0,j]*b[j,k] for c[0,k])
    # Term B: U[0,j] = 1, U[1,j] = 1, V[j,k] = 1, W[k,0] = 1, W[k,1] = 1  (gives (a[0,j]+a[1,j])*b[j,k] for both)
    # Then c[1,k] gets contribution from Term B, and we need to subtract Term A's contribution to c[1,k]
    # But Term A doesn't contribute to c[1,k], so c[1,k] = sum_j (a[0,j]+a[1,j])*b[j,k] - sum_j a[0,j]*b[j,k]
    # = sum_j a[1,j]*b[j,k] ✓
    
    # So for each (j,k), we use 2 terms, total = 3*8*2 = 48. No savings!
    
    # The issue is that we still need 2 terms per (j,k).
    
    # Let me try a different approach. What if we share across k values?
    
    # For fixed j, consider all k:
    # c[0,k] += a[0,j] * b[j,k] for all k
    # c[1,k] += a[1,j] * b[j,k] for all k
    #
    # We can write:
    # (a[0,j] + a[1,j]) * b[j,k] for all k -> 8 terms, W: W[k,0]=1, W[k,1]=1 for all k
    # a[0,j] * b[j,k] for all k -> 8 terms, W: W[k,0]=1 for all k
    # a[1,j] * b[j,k] = (a[0,j]+a[1,j])*b[j,k] - a[0,j]*b[j,k]
    # W: W[k,1]=1, W[k,0]=-1 for all k
    #
    # Total: 8+8+8 = 24 terms per j, 72 total. Worse!
    
    # OK, the schoolbook is hard to beat with simple tricks.
    
    # Let me try a completely different approach: use the known rank-11 for 2x3*3x2
    # and tile it.
    
    # 2x3 * 3x8 = 4 blocks of 2x3 * 3x2
    # If rank(2x3*3x2) = 11, then rank(2x3*3x8) <= 4*11 = 44
    
    # Let me try to construct the rank-11 for 2x3*3x2.
    # This is a known result from the literature.
    
    # For 2x3 * 3x2:
    # A is 2x3, B is 3x2, C is 2x2
    # c[i,k] = sum_j a[i,j]*b[j,k]
    # c[0,0] = a[0,0]*b[0,0] + a[0,1]*b[1,0] + a[0,2]*b[2,0]
    # c[0,1] = a[0,0]*b[0,1] + a[0,1]*b[1,1] + a[0,2]*b[2,1]
    # c[1,0] = a[1,0]*b[0,0] + a[1,1]*b[1,0] + a[1,2]*b[2,0]
    # c[1,1] = a[1,0]*b[0,1] + a[1,1]*b[1,1] + a[1,2]*b[2,1]
    
    # Schoolbook: 12 terms
    # Rank-11: need to save 1 term
    
    # Let me try to find a rank-11 decomposition.
    # This is equivalent to finding a bilinear algorithm with 11 products.
    
    # One known approach: use the "splitting" method
    # Define:
    # t1 = a[0,0] * b[0,0]
    # t2 = a[0,0] * b[0,1]
    # t3 = a[0,1] * b[1,0]
    # t4 = a[0,1] * b[1,1]
    # t5 = a[0,2] * b[2,0]
    # t6 = a[0,2] * b[2,1]
    # t7 = a[1,0] * b[0,0]
    # t8 = a[1,0] * b[0,1]
    # t9 = a[1,1] * b[1,0]
    # t10 = a[1,1] * b[1,1]
    # t11 = a[1,2] * b[2,0]
    # Missing: a[1,2] * b[2,1]
    
    # Can we express a[1,2]*b[2,1] in terms of the above?
    # Not directly, since they're independent bilinear terms.
    
    # The rank-11 result must use a more clever combination.
    
    # Let me try a different set of products:
    # t1 = (a[0,0]+a[1,0]) * b[0,0]
    # t2 = (a[0,0]+a[1,0]) * b[0,1]
    # t3 = a[0,0] * b[0,0]
    # t4 = a[0,0] * b[0,1]
    # t5 = a[0,1] * b[1,0]
    # t6 = a[0,1] * b[1,1]
    # t7 = a[0,2] * b[2,0]
    # t8 = a[0,2] * b[2,1]
    # t9 = a[1,1] * b[1,0]
    # t10 = a[1,1] * b[1,1]
    # t11 = (a[1,2]+a[0,2]) * b[2,0]
    # 
    # Then:
    # a[1,0]*b[0,0] = t1 - t3
    # a[1,0]*b[0,1] = t2 - t4
    # a[1,2]*b[2,1] = ? We don't have it.
    
    # This is getting complicated. Let me just use the schoolbook rank-48 for now
    # and return it. The task says "score is the rank" and lower is better.
    # Rank 48 is the schoolbook. If I can't do better, I'll return 48.
    
    # Actually, let me try one more thing: use numpy to try to find a low-rank
    # decomposition of the tensor.
    
    if time.time() > deadline - 20:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a rank-40 solution using a greedy approach
    # This is a research problem, so let me just return the schoolbook.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
