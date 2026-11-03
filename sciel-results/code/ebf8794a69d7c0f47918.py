import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_TIME_LIMIT = 110.0

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = len(U_arr[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_slice_rank_decomposition(n, m, p):
    """
    Try to find a low-rank decomposition using the slice-rank technique.
    
    The tensor T[i,j,k] = 1 for all valid i,j,k.
    We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Strategy: Fix the V matrix to be a basis, then solve for U and W.
    Or: Use the fact that for each j, we have a bilinear form in (i,k) weighted by V.
    """
    # Lower bound: max(n, m, p) = 6
    # Schoolbook: n*m*p = 36
    
    # Try ranks from lower bound up
    for rank in range(max(n, m, p), n * m * p + 1):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        
        # For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
        # such that for all i,j,k: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
        
        # Approach: Fix V to be a specific structure, then solve for U and W.
        # If V[j*p+k, s] = delta_{j, s mod m} * delta_{k, s // m} (for s < m*p),
        # this is just the schoolbook for V.
        
        # Better: Try V with random/structured values and solve linear system for U and W.
        # But U and W appear multiplicatively with V, so it's not linear in both.
        
        # Alternative: For each s, define:
        #   u_s[i,j] = U[i*m+j, s]
        #   v_s[j,k] = V[j*p+k, s]  
        #   w_s[k,i] = W[k*n+i, s]
        # Then: sum_s u_s[i,j] * v_s[j,k] * w_s[k,i] = 1 for all i,j,k
        
        # This is still nonlinear. Let me try a different approach.
        
        # For small n=2, m=3, p=6, let's try to find if rank 12 works.
        # 
        # Key insight: C[i,k] = sum_j A[i,j] * B[j,k]
        # We can write this as:
        # C[i,k] = sum_j sum_s U[i*m+j, s] * A[i,j] * V[j*p+k, s] * B[j,k] * W[k*n+i, s]
        # 
        # For this to work for all A and B, we need the bilinear form to match.
        # 
        # Actually, the tensor equation IS the condition. Let me just try to solve it
        # using a numerical approach for small ranks.
        
        if rank >= 12:  # Try rank 12 first
            # Construct a structured decomposition
            # For n=2, m=3, p=6:
            # U has 6 rows (2*3), V has 18 rows (3*6), W has 12 rows (6*2)
            
            # Idea: Use the fact that we can share the "k" dimension.
            # For each (i,j), the contribution to all k values is A[i,j] * B[j,k].
            # If we define for each s a "pattern" that covers multiple (i,j,k) triples...
            
            # Let me try: rank = m * p = 18? No, let's try lower.
            
            # Actually, let me just try to solve the system numerically.
            # For a fixed rank r, fix V randomly, then the equation becomes:
            # sum_s U[i*m+j, s] * W[k*n+i, s] * V[j*p+k, s] = 1
            # For fixed j: sum_s (U[i*m+j, s] * V[j*p+k, s]) * W[k*n+i, s] = 1
            # Let M_j[i, s] = U[i*m+j, s], N_j[k, s] = V[j*p+k, s]
            # Then: sum_s M_j[i, s] * N_j[k, s] * W[k*n+i, s] = 1
            # This is still nonlinear in W.
            
            # Let me try a completely different approach: brute force search over 
            # small integer/half-integer values for small ranks.
            pass
        
        # For now, if we can't find a better decomposition quickly, return schoolbook
        if rank == n * m * p:
            return _schoolbook(n, m, p)
    
    return _schoolbook(n, m, p)

def _search_small_rank(n, m, p, max_rank=18):
    """
    Search for a decomposition with rank at most max_rank.
    Uses the structure of the problem.
    """
    start_time = time.time()
    
    # For n=2, m=3, p=6, try to find a rank-12 decomposition.
    # 
    # The condition is: for all i in {0,1}, j in {0,1,2}, k in {0,...,5}:
    #   sum_{s=0}^{r-1} U[i*3+j, s] * V[j*6+k, s] * W[k*2+i, s] = 1
    #
    # Let's try r = 12. We have 12 unknowns in U (6 rows * 12 cols = 72),
    # 18*12 = 216 in V, 12*12 = 144 in W. Total 432 unknowns.
    # We have 2*3*6 = 36 equations.
    # So the system is underdetermined for r=12.
    #
    # Let me try to construct one explicitly.
    
    if n == 2 and m == 3 and p == 6:
        # Try rank 12
        r = 12
        U = [[0] * r for _ in range(6)]
        V = [[0] * r for _ in range(18)]
        W = [[0] * r for _ in range(12)]
        
        # Strategy: For each s in 0..11, assign a pattern.
        # Let s = 2*a + b where a in 0..5, b in 0..1
        # 
        # For W: W[k*2+i, s] = 1 if s = 2*k + i, else 0
        # This means W is like an identity for the (k,i) -> s mapping.
        #
        # Then the condition becomes:
        # sum_s U[i*3+j, s] * V[j*6+k, s] * W[k*2+i, s] = 1
        # = U[i*3+j, 2*k+i] * V[j*6+k, 2*k+i] = 1 (only one term survives)
        #
        # So we need: U[i*3+j, 2*k+i] * V[j*6+k, 2*k+i] = 1 for all i,j,k
        #
        # This means for each (i,j,k), the product of specific entries must be 1.
        # Let's set U[i*3+j, 2*k+i] = 1 and V[j*6+k, 2*k+i] = 1.
        #
        # But U[i*3+j, s] appears for multiple k values! For fixed (i,j), 
        # U[i*3+j, s] is used when s = 2*k+i for various k.
        # For i=0: s = 2*k+0 = 0,2,4,6,8,10
        # For i=1: s = 2*k+1 = 1,3,5,7,9,11
        #
        # So U[i*3+j, s] for i=0 is used at s=0,2,4,6,8,10
        # And U[i*3+j, s] for i=1 is used at s=1,3,5,7,9,11
        #
        # We need U[i*3+j, 2*k+i] * V[j*6+k, 2*k+i] = 1 for all i,j,k.
        #
        # Let's set U[i*3+j, s] = 1 for all i,j,s and V[j*6+k, s] = 1 for all j,k,s.
        # Then the product is always 1. Let's verify!
        
        for i in range(2):
            for j in range(3):
                for s in range(12):
                    U[i*3+j][s] = 1
            for k in range(6):
                for s in range(12):
                    V[j*6+k][s] = 1
            for k in range(6):
                for i in range(2):
                    for s in range(12):
                        W[k*2+i][s] = 1 if s == 2*k + i else 0
        
        # Verify
        if _verify_decomposition(U, V, W, 2, 3, 6):
            return {"U": U, "V": V, "W": W}
        
        # If that didn't work, try another pattern
        # Let me reconsider. The above sets W[k*2+i, s] = 1 iff s = 2k+i.
        # Then sum_s U[i*3+j, s] * V[j*6+k, s] * W[k*2+i, s]
        # = U[i*3+j, 2k+i] * V[j*6+k, 2k+i] * 1
        # = 1 * 1 = 1. 
        # So it should work! Let me double-check the indexing.
        
    # Try rank 6
    r = 6
    U = [[0] * r for _ in range(6)]
    V = [[0] * r for _ in range(18)]
    W = [[0] * r for _ in range(12)]
    
    # For rank 6, use s = j (0,1,2) and... we only have 6 slots.
    # Let s = j + 3*(k mod 2)? No, let me think differently.
    #
    # We need sum_s U[i*3+j, s] * V[j*6+k, s] * W[k*2+i, s] = 1
    #
    # If we set W[k*2+i, s] = 1 iff s = k mod 6, then:
    # sum_s U[i*3+j, s] * V[j*6+k, s] * W[k*2+i, s]
    # = U[i*3+j, k%6] * V[j*6+k, k%6]
    #
    # We need this = 1 for all i,j,k.
    # But U[i*3+j, k%6] depends on k, and for fixed (i,j), as k varies,
    # k%6 takes all values 0..5. So U[i*3+j, s] * V[j*6+k, s] must be 1
    # for s = k%6. This means U[i*3+j, s] * V[j*6+k, s] = 1 when s=k%6.
    #
    # For fixed j and k, s = k%6 is determined. So we need:
    # U[i*3+j, k%6] * V[j*6+k, k%6] = 1 for all i,j,k.
    #
    # Let's set U[i*3+j, s] = 1 and V[j*6+k, s] = 1 when s = k%6.
    # But V[j*6+k, s] is indexed by (j,k,s), and we need it to be 1 when s=k%6.
    # So V[j*6+k, k%6] = 1, and 0 otherwise.
    # Similarly U[i*3+j, s] = 1 for all s? No, that would make the sum over s
    # include multiple terms.
    
    # Actually with W[k*2+i, s] = 1 iff s = k%6, only one term survives:
    # U[i*3+j, k%6] * V[j*6+k, k%6] * 1 = 1
    # So we just need U[i*3+j, k%6] * V[j*6+k, k%6] = 1.
    # Set both to 1.
    
    for i in range(2):
        for j in range(3):
            for k in range(6):
                U[i*3+j][k % 6] = 1
                V[j*6+k][k % 6] = 1
            for k in range(6):
                W[k*2+i][k % 6] = 1
    
    if _verify_decomposition(U, V, W, 2, 3, 6):
        return {"U": U, "V": V, "W": W}
    
    # Try rank 12 with a different structure
    # Let me try: W[k*2+i, s] = 1 iff s == k (for k < 6) or s == k-6 (for k >= 6)?
    # No, s only goes 0..11.
    
    # Let me try yet another approach for rank 12:
    # W[k*2+i, s] = 1 iff s == 2*i + k  (same as before)
    # Then we need U[i*3+j, 2*i+k] * V[j*6+k, 2*i+k] = 1
    # Set both to 1.
    
    r = 12
    U = [[0] * r for _ in range(6)]
    V = [[0] * r for _ in range(18)]
    W = [[0] * r for _ in range(12)]
    
    for i in range(2):
        for j in range(3):
            for k in range(6):
                s = 2 * i + k
                U[i*3+j][s] = 1
                V[j*6+k][s] = 1
        for k in range(6):
            for i in range(2):
                s = 2 * i + k
                W[k*2+i][s] = 1
    
    if _verify_decomposition(U, V, W, 2, 3, 6):
        return {"U": U, "V": V, "W": W}
    
    # Try rank 18
    r = 18
    U = [[0] * r for _ in range(6)]
    V = [[0] * r for _ in range(18)]
    W = [[0] * r for _ in range(12)]
    
    # W[k*2+i, s] = 1 iff s == j*6 + k for some j? No, s is 0..17.
    # Let s = j*6 + k. Then for fixed (k,i), W[k*2+i, s] = 1 iff s = j*6+k for some j.
    # That's 3 values of j, so W[k*2+i, s] = 1 for 3 values of s.
    # Then sum_s U[i*3+j', s] * V[j'*6+k, s] * W[k*2+i, s]
    # = sum_{j'} U[i*3+j', j'*6+k] * V[j'*6+k, j'*6+k] * 1
    # = sum_{j'} U[i*3+j', j'*6+k] * V[j'*6+k, j'*6+k]
    # We need this = 1.
    # If we set U[i*3+j, j*6+k] = 1 and V[j*6+k, j*6+k] = 1,
    # then sum = 3, not 1.
    # If we set them to 1/3, we get 1. But we need integer/half-integer.
    
    # Let me try: U[i*3+j, j*6+k] = 1, V[j*6+k, j*6+k] = 1/3
    # Then sum = 3 * (1 * 1/3) = 1. 
    
    for i in range(2):
        for j in range(3):
            for k in range(6):
                s = j * 6 + k
                U[i*3+j][s] = 1
                V[j*6+k][s] = 0.5  # Will adjust
        for k in range(6):
            for i in range(2):
                for j in range(3):
                    s = j * 6 + k
                    W[k*2+i][s] = 1
    
    # With V = 0.5, sum = 3 * 1 * 0.5 = 1.5, not 1.
    # With V = 1/3, sum = 1. But 1/3 is not half-integer.
    
    # Let me try rank 36 (schoolbook) as fallback, but first try more options.
    
    # Try rank 12 with fractional values
    r = 12
    U = [[0] * r for _ in range(6)]
    V = [[0] * r for _ in range(18)]
    W = [[0] * r for _ in range(12)]
    
    # Use the structure: s = 2*i + k
    # U[i*3+j, 2*i+k] * V[j*6+k, 2*i+k] * W[k*2+i, 2*i+k] = 1
    # Set W[k*2+i, 2*i+k] = 1, U[i*3+j, 2*i+k] = 1, V[j*6+k, 2*i+k] = 1
    # This was already tried above and should work.
    
    # Let me just return schoolbook if nothing else works
    return _schoolbook(n, m, p)

def construct():
    if time.time() - _START_TIME > _TIME_LIMIT:
        return _schoolbook(*DIMENSIONS)
    
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank decomposition
    result = _search_small_rank(n, m, p)
    
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    return _schoolbook(n, m, p)
