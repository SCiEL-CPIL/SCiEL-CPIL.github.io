"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    # Deadline guard
    deadline = time.time() + 120.0
    
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_school = n * m * p  # 3*4*7 = 84
    
    # Build the multiplication tensor T[i, j, k] = delta_{i,j,k}
    # Actually the tensor is T[i, j, k] where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor has indices (i, j, k) with T[i,j,k] = 1 if it contributes to C[i,k]
    # More precisely, the tensor T is in R^{n x m x p} where T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # Wait, let me re-read the task.
    # "exactly decompose the matrix multiplication tensor"
    # The tensor is T in R^{(n*m) x (m*p) x (p*n)}
    # T[(i,j), (j',k), (k',i')] = 1 if i=i' and j=j' and k=k'
    # So T is a 3-tensor of size (n*m) x (m*p) x (p*n)
    
    # Let's use a greedy/rank-reduction approach.
    # We start with the schoolbook decomposition and try to reduce rank.
    
    # First, let's try to find a good rank via random linear algebra.
    # The tensor T can be flattened to a matrix of size (n*m) x (m*p*(p*n))
    # Or (n*m*(m*p)) x (p*n)
    # The rank of the tensor is the minimum r such that T = sum_{c=1}^r u_c v_c^T w_c (outer product)
    
    # Let's use the approach of trying random rank-r decompositions and checking if they work.
    # But that's hard. Instead, let's use known results.
    
    # For 3x4x7, the schoolbook is 84.
    # Strassen-like methods can reduce this.
    # Let's try a systematic search for rank reduction.
    
    # Method: Start with schoolbook, then try to combine terms.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be combined if they share structure.
    
    # Actually, let's use a different approach: random linear combinations.
    # We can try to find a rank-r decomposition by solving a system of equations.
    
    # Let's try ranks from a lower bound up to schoolbook.
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    
    # For small matrices, we can try to use the fact that the tensor is sparse.
    
    # Let's implement a simple rank-reduction algorithm:
    # 1. Start with schoolbook terms.
    # 2. Try to merge pairs of terms that have overlapping support.
    
    # Actually, let's try a more principled approach using linear algebra.
    # The tensor T can be written as a sum of rank-1 tensors.
    # We want to find the minimum number of such terms.
    
    # For now, let's try to find a rank-28 decomposition (the trivial lower bound).
    # If that fails, we'll try higher ranks.
    
    # Let's use a random approach: generate random rank-r decompositions and check.
    # But checking is expensive. Instead, let's use the fact that for specific small sizes,
    # there might be known optimal decompositions.
    
    # Let's try to construct a decomposition using block methods.
    # 3x4x7: We can think of A as 3x4, B as 4x7.
    # Split B into two parts: B1 (4x3) and B2 (4x4).
    # Then C = A*B1 + A*B2.
    # A*B1 is 3x3, A*B2 is 3x4.
    # Rank of 3x3x4: schoolbook 36, but can be reduced.
    # Rank of 3x4x4: schoolbook 48.
    # This doesn't immediately help.
    
    # Let's try a different split.
    # Actually, let's just use a greedy algorithm to reduce rank.
    
    # Start with schoolbook.
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k))
    
    rank = len(terms)
    
    # Try to merge terms.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if:
    # i1==i2 and j1==j2, then we can combine k1 and k2 into a single term with coefficients.
    # Actually, that's not right. The tensor is T[i,j,k] = delta(i,j,k).
    # A rank-1 term is u_i v_j w_k where u, v, w are vectors.
    # The schoolbook term for (i,j,k) has u=e_i, v=e_j, w=e_k.
    
    # To reduce rank, we need to find linear dependencies.
    # This is equivalent to finding the tensor rank, which is NP-hard in general.
    
    # For this specific case, let's try a few known techniques.
    
    # Technique 1: Use the fact that 7 = 4 + 3.
    # Write B = [B1 | B2] where B1 is 4x4 and B2 is 4x3.
    # C = A*B1 + A*B2.
    # Compute A*B1 (3x4) and A*B2 (3x3) separately.
    # Rank of 3x4x4: Let's see if we can do better than 48.
    # Rank of 3x4x3: Let's see if we can do better than 36.
    
    # Actually, this approach requires computing the ranks of subproblems,
    # which is as hard as the original problem.
    
    # Let's try a random restart approach with a time limit.
    best_rank = rank_school
    best_decomp = None
    
    # Generate schoolbook decomposition
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_decomp = (U_school, V_school, W_school)
    
    # Try to reduce rank by random linear combinations.
    # We'll try ranks from max(n*m, m*p, p*n) to rank_school.
    min_rank = max(n*m, m*p, p*n)
    
    for r in range(min_rank, rank_school + 1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random attempts for this rank
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Random rank-r decomposition
            U_rand = np.random.randn(n*m, r)
            V_rand = np.random.randn(m*p, r)
            W_rand = np.random.randn(p*n, r)
            
            # Check if this decomposition is valid.
            # The condition is that for all i, j, k:
            # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if (i,j,k) is a valid term, 0 otherwise.
            # Actually, the tensor T[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
            # Wait, no. The multiplication tensor is defined such that:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # So the tensor T has T[i,j,k] = 1 for all i,j,k (since each product A[i,j]*B[j,k] contributes to C[i,k]).
            # Actually, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
            # So we need: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k.
            
            # Let's check this condition.
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c] for c in range(r))
                        if abs(val - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # If valid, we found a rank-r decomposition.
            best_rank = r
            best_decomp = (U_rand, V_rand, W_rand)
            break  # No need to try higher ranks for this r
        
        if best_rank == r:
            break  # Found the minimum rank
    
    # Convert to the required format
    U, V, W = best_decomp
    result_U = [[int(round(U[i, c] * 2)) / 2 for c in range(best_rank)] for i in range(n*m)]
    result_V = [[int(round(V[i, c] * 2)) / 2 for c in range(best_rank)] for i in range(m*p)]
    result_W = [[int(round(W[i, c] * 2)) / 2 for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": result_U, "V": result_V, "W": result_W}
# EVOLVE-BLOCK-END
