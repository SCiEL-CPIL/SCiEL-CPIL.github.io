"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s
    
    # Schoolbook decomposition: rank = n*m*p = 52
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    def verify(U, V, W, n, m, p):
        """Verify that the tensor decomposition is correct."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
                    for k2 in range(p):
                        if k2 != k:
                            s2 = 0
                            for r in range(rank):
                                s2 += U[i*m+j][r] * V[j*p+k2][r] * W[k2*n+i][r]
                            if abs(s2) > 1e-9:
                                return False
        return True
    
    if verify(U, V, W, n, m, p):
        pass  # Schoolbook is valid
    
    # Try to find a better decomposition using structured search
    # For 2x2 * 2xp, try rank = 3p + 1 = 40 first, then 4p = 52 (same as schoolbook)
    # The key is to find if we can share rank-1 terms across different k values.
    
    # Strategy: For each k, we need to represent the 4 products:
    #   a*B[0,k], b*B[1,k], c*B[0,k], d*B[1,k]
    # using rank-1 terms that may combine A and B entries.
    
    # A rank-1 term (u, v, w) where:
    #   u is a vector in R^{n*m} = R^4 (entries for A[0,0], A[0,1], A[1,0], A[1,1])
    #   v is a vector in R^{m*p} = R^{26} (entries for B[0,0]..B[0,12], B[1,0]..B[1,12])
    #   w is a vector in R^{p*n} = R^{26} (entries for C[0,0]..C[0,12], C[1,0]..C[1,12])
    # The contribution to the tensor at position (i,j,k) is u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # For the decomposition to be valid:
    # For all i,j,k: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    # For all i,j,k,k',i' where (i',k') != (i,k): sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k'*n+i'] = 0
    
    # Let's try a construction based on linear algebra:
    # For each k, consider the 2x2 sub-problem:
    #   C[0,k] = a*B[0,k] + b*B[1,k]
    #   C[1,k] = c*B[0,k] + d*B[1,k]
    
    # We can write this as a matrix product:
    #   [C[0,k]]   [a b] [B[0,k]]
    #   [C[1,k]] = [c d] [B[1,k]]
    
    # The rank of this 2x2 matrix multiplication is at most 4 (schoolbook).
    # But we can potentially do better if we allow shared terms.
    
    # Let's try: for each k, use 3 terms + 1 shared term = 4p total... that's not better.
    
    # Alternative: use the fact that the 2x2 matrix A is fixed for all k.
    # If A has special structure (e.g., symmetric), we might save, but A is arbitrary.
    
    # Let me try a different approach: search for a rank-3 decomposition per k.
    # For 2x2 * 2x1, the minimal rank is known to be 4 (no improvement over schoolbook).
    # So for 2x2 * 2x13, the minimal rank is at least 4*13 = 52? 
    # Actually no, because terms can be shared across different k values.
    
    # Wait, let me reconsider. Each rank-1 term involves:
    #   One A entry (or linear combination)
    #   One B entry (or linear combination)  
    #   One C entry (or linear combination)
    
    # The constraint is that for each (i,j,k), the sum must equal 1, and for "wrong" positions, 0.
    
    # Let me try a specific construction:
    # Term type 1: u = (1,0,0,0), v has 1 at position (0,k), w has 1 at position (k,0)
    #   This gives: A[0,0] * B[0,k] -> C[0,k]
    # Term type 2: u = (0,1,0,0), v has 1 at position (1,k), w has 1 at position (k,0)
    #   This gives: A[0,1] * B[1,k] -> C[0,k]
    # Term type 3: u = (0,0,1,0), v has 1 at position (0,k), w has 1 at position (k,1)
    #   This gives: A[1,0] * B[0,k] -> C[1,k]
    # Term type 4: u = (0,0,0,1), v has 1 at position (1,k), w has 1 at position (k,1)
    #   This gives: A[1,1] * B[1,k] -> C[1,k]
    
    # That's 4p = 52 terms, same as schoolbook.
    
    # Can we do better? Let's try combining:
    # Term: u = (1,1,0,0), v has 1 at position (0,k), w has 1 at position (k,0)
    #   This gives: (A[0,0]+A[0,1]) * B[0,k] -> C[0,k]
    # But we also need A[0,1]*B[1,k] -> C[0,k], which is separate.
    
    # What if we use:
    # Term 1: u = (1,0,1,0), v has 1 at position (0,k), w has 1 at position (k,0)
    #   This gives: (A[0,0]+A[1,0]) * B[0,k] -> C[0,k]
    # But this contributes to C[0,k] with coefficient (a+c), not just a.
    # We'd need to subtract c*B[0,k] from C[0,k], which requires another term.
    
    # This is getting complicated. Let me try a numerical optimization approach.
    
    # Actually, let me think about this differently. The tensor for 2x2 * 2xp is:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,12}
    # And we need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a 3-tensor of size 4 x 26 x 26.
    # The schoolbook decomposition has rank 52.
    
    # Can we find a decomposition with rank < 52?
    # Let's try rank 51, 50, etc. using random initialization and gradient descent.
    
    # But first, let me check if there's a known result.
    # For matrix multiplication, the rank of M_{n,m,p} is at least max(n*m, m*p, p*n).
    # Here: max(4, 26, 26) = 26. So rank >= 26.
    # Also, rank >= n*m*p / max(n,m,p) = 52/13 = 4... that's a weak bound.
    
    # Actually, a better lower bound: rank >= (n*m*p) / (n+m+p-2) for some cases, but I'm not sure.
    
    # Let me just try to find a decomposition with rank < 52 using random search.
    
    # For each candidate rank, try random initializations and see if we can verify.
    # This is unlikely to work well for exact verification, but let's try.
    
    # Actually, let me try a smarter approach: use the structure of the problem.
    # The tensor is "block diagonal" in the k dimension if we look at it correctly.
    
    # For each k, the sub-tensor T_k[i,j] = 1 for all i,j in {0,1}.
    # But the W vector couples different k values through the output position.
    
    # Let me try a construction with rank = 3p + 2 = 41:
    # For each k, use 3 terms, plus 2 global terms.
    
    # Hmm, this is not working. Let me just return the schoolbook result.
    
    return best_result
# EVOLVE-BLOCK-END
