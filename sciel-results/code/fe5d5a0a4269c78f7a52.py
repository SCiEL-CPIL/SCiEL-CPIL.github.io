"""Combined: structural Strassen-style construction for 3x4x8 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-reduced exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x8 matrix.
    
    The tensor T has entries T[i, j, k] for i in [0,n), j in [0,m), k in [0,p).
    A rank-r decomposition gives U (n*m x r), V (m*p x r), W (p*n x r) such that:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For matrix multiplication A (n x m) * B (m x p) = C (n x p):
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k.
    
    Strategy: Use Strassen's algorithm on the j-dimension (size 4 = 2+2).
    For each pair of j-values (j0, j1), the contribution A[i,j0]*B[j0,k] + A[i,j1]*B[j1,k]
    can be computed with 7 terms instead of 4 using Strassen-like identities,
    but we need to be careful about how we share across i and k.
    
    Actually, for a 2x2x2 block, Strassen reduces 8 to 7. For our problem:
    - j-dimension is 4, split into two pairs: {0,1} and {2,3}
    - For each pair, for each (i,k), we have a 2-term dot product
    - The key insight: we can use the Strassen decomposition on the j-pair
      to reduce the number of multiplications per (i,k) pair.
    
    Let's think more carefully. The full tensor is:
      T[i,j,k] = 1 for all i in [0,3), j in [0,4), k in [0,8)
    
    We can write this as:
      T = T_pair1 + T_pair2
    where T_pair1[i,j,k] = 1 if j in {0,1}, else 0
    and T_pair2[i,j,k] = 1 if j in {2,3}, else 0
    
    For T_pair1, we need to decompose:
      sum_{j in {0,1}} A[i,j] * B[j,k] for each (i,k)
    
    This is a 3x2x8 tensor. The schoolbook rank is 3*2*8 = 48.
    Using Strassen on the j-dimension (size 2), we can potentially reduce this.
    
    Actually, Strassen's algorithm is for 2x2x2 matrices. For a 3x2x8 tensor,
    we can view it as 3*8 independent 1x2x1 tensors, which don't benefit from Strassen.
    
    Let me try a different approach. The key realization is that for matrix multiplication,
    the rank of the tensor M_{n,m,p} is at most n*m*p, but can be lower.
    
    For 3x4x8, let's try to construct a decomposition with rank < 96.
    
    One approach: use the fact that we can combine terms with the same (i,k) but different j.
    For fixed i and k, C[i,k] = sum_j A[i,j] * B[j,k].
    
    If we define:
      S_i = A[i,0] + A[i,1]
      D_i = A[i,0] - A[i,1]
      S_k = B[0,k] + B[1,k]
      D_k = B[0,k] - B[1,k]
    
    Then:
      A[i,0]*B[0,k] + A[i,1]*B[1,k] = (S_i * S_k + D_i * D_k) / 2
    
    This uses 2 terms instead of 2, so no gain. But if we can share S_i and S_k across multiple k or i values, we might gain.
    
    Actually, the gain comes from the fact that S_i is independent of k, and S_k is independent of i.
    So we can define:
      For each i: S_i = A[i,0] + A[i,1] (3 values)
      For each k: S_k = B[0,k] + B[1,k] (8 values)
    
    Then S_i * S_k contributes to C[i,k] for the j-pair {0,1}.
    Similarly for D_i * D_k.
    
    So for the j-pair {0,1}, we need:
      - 3 terms for S_i (one per i)
      - 8 terms for S_k (one per k)
      - But these are not independent; they're combined as S_i * S_k
    
    In the tensor decomposition, this means we need terms that factor as:
      U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the S_i * S_k term, we'd need:
      U[i*4+0, c] = 0.5 (for j=0), U[i*4+1, c] = 0.5 (for j=1)
      V[0*8+k, c] = 1, V[1*8+k, c] = 1
      W[k*3+i, c] = 1
    
    But this would give 0.5 * 1 * 1 = 0.5 for each (i,j,k) with j in {0,1}, which is not 1.
    
    Let me reconsider. The correct decomposition for the j-pair {0,1} is:
      A[i,0]*B[0,k] + A[i,1]*B[1,k]
      = (A[i,0]+A[i,1])*(B[0,k]+B[1,k])/2 + (A[i,0]-A[i,1])*(B[0,k]-B[1,k])/2
    
    So we need two sets of terms:
      Set 1: U[i*4+j, c] = 0.5 for j in {0,1}, V[j*8+k, c] = 1 for j in {0,1}, W[k*3+i, c] = 1
      Set 2: U[i*4+j, c] = 0.5 for j=0, -0.5 for j=1, V[0*8+k, c] = 1, V[1*8+k, c] = -1, W[k*3+i, c] = 1
    
    But this still uses 2 terms per (i,k), so no gain.
    
    The gain would come if we could share the U and V factors across multiple (i,k) pairs.
    But in the tensor decomposition, each term is a rank-1 tensor, so we can't directly share.
    
    Let me try a completely different approach: use a known lower bound and try to construct a decomposition that achieves it.
    
    For 3x4x8, the schoolbook rank is 96. The question is whether we can do better.
    
    Actually, I recall that for matrix multiplication, the rank of M_{n,m,p} is at least max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    But this is a weak lower bound.
    
    A better lower bound comes from the slice rank. The slice rank of M_{n,m,p} is at least max(n, m, p) = 8.
    But the rank is at least the slice rank, so rank >= 8. This is also weak.
    
    Let me try to construct a decomposition with rank 80 by using the following idea:
    - For each i in [0,3), k in [0,8), the contribution is sum_j A[i,j]*B[j,k].
    - This is a 4-dimensional dot product.
    - We can use a 4x4 matrix M such that M[i,j] * M[j,k] = delta_{i,k} (Kronecker delta).
    - But this requires M to be invertible, which is possible for 4x4.
    
    Actually, let's just return the schoolbook decomposition as a valid baseline, and try to improve it if we have time.
    
    For now, let's construct the schoolbook decomposition with rank 96.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 96
    
    # Start with schoolbook
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Check time budget
    if time.time() > deadline:
        return best_result
    
    # Try to find a lower-rank decomposition using a more sophisticated method.
    # For 3x4x8, I'll attempt to use the following construction:
    #
    # View the 4x8 matrix multiplication as two 2x8 multiplications.
    # For each 2x8 multiplication, use a Winograd-like approach.
    #
    # For a 2x8 matrix A (2 rows, 8 cols) times a 8x8 matrix B, the result
    # is 2x8. The standard approach uses 2*8*8=128 multiplications.
    # Winograd can reduce this, but for our tensor decomposition, we need
    # a different approach.
    
    # Let me try a random search for a lower-rank decomposition.
    # For rank r, we need to find U (12 x r), V (32 x r), W (24 x r) such that
    # for all i, j, k: sum_c U[i*4+j, c] * V[j*8+k, c] * W[k*3+i, c] = 1
    
    # This is a system of 96 equations. We can try to solve it with r < 96
    # using a randomized approach.
    
    # For each candidate rank r, try to find a solution.
    # Start with r = 88 and go down.
    
    for target_rank in [88, 84, 80, 76, 72]:
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition with this rank
        # Use a simple iterative approach: initialize randomly, then refine
        
        success = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize U, V, W with small random values
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            # Set up the system: for each (i,j,k), we need
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            #
            # This is nonlinear, so we'll use a different approach.
            # Instead, fix U and V, then solve for W, and iterate.
            
            # Initialize U and V with identity-like patterns
            for c in range(target_rank):
                # Assign each column to a specific (i,j,k) triple
                idx = c % (n * m * p)
                i = idx // (m * p)
                rem = idx % (m * p)
                j = rem // p
                k = rem % p
                
                U_try[i * m + j, c] = 1
                V_try[j * p + k, c] = 1
                W_try[k * n + i, c] = 1
            
            # Check if this works (it should for rank >= 96, but we're trying lower)
            # For rank < 96, some (i,j,k) triples won't have their own column,
            # so they'll get contributions from other columns.
            
            # Check correctness
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U_try[i * m + j, c] * V_try[j * p + k, c] * W_try[k * n + i, c]
                        if abs(s - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to integer/half-integer if possible
                success = True
                U_final = [[float(x) for x in U_try[row]] for row in range(n * m)]
                V_final = [[float(x) for x in V_try[row]] for row in range(m * p)]
                W_final = [[float(x) for x in W_try[row]] for row in range(p * n)]
                best_result = {"U": U_final, "V": V_final, "W": W_final}
                best_rank = target_rank
                break
        
        if success:
            break
    
    return best_result
# EVOLVE-BLOCK-END
