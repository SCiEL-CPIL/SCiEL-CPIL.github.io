"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Try to find a lower-rank decomposition using structured search
    # Start with schoolbook rank 54, try to reduce
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Check if the decomposition exactly reproduces matrix multiplication."""
        rank = len(U[0])
        # Build the multiplication tensor from U, V, W
        # T[i*m+j, j'*p+k, k'*n+i'] should be sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k'*n+i',c]
        # This should equal 1 iff j==j' and i==i' (for the same k... wait)
        # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: T_{i,j; j',k; i',k'} = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        # Using trace convention: W index is k*n+i, so W[k*n+i]
        # T[a,b,c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
        # T[a,b,c] = 1 iff j==j' and i==i' and k==k'
        # i.e., a % m == b // p, a // m == c % n, b % p == c // n
        
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            for k2 in range(p):
                                a = i * m + j
                                b = j2 * p + k
                                c = k2 * n + i2
                                expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                                actual = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    best = schoolbook()
    
    # Try a rank-50 construction based on algebraic identities
    # For 2x3 x 3x9, try to exploit that n=2
    # Idea: use the identity for 2x2x2 Strassen-like reduction on subsets
    
    # Let's try: for each pair of columns (k1, k2) in the output, 
    # see if we can save rank by combining computations
    # This is complex; let's try a direct rank-reduction search
    
    if time.time() < deadline:
        # Try random rank-reduction: start with schoolbook, try to zero out some entries
        # and see if the decomposition still holds
        rank_54 = n * m * p
        U54 = np.array(best["U"], dtype=np.float64)
        V54 = np.array(best["V"], dtype=np.float64)
        W54 = np.array(best["W"], dtype=np.float64)
        
        # Try to find linear dependencies among the rank-1 terms
        # Each term c has U[:,c], V[:,c], W[:,c]
        # If we can express one term as a combination of others, we can reduce rank
        # This is a hard problem; skip for now
        
        # Alternative: try known constructions
        # For 2x3 x 3x9, one approach: treat as 3 separate 2x3 x 3x3
        # For 2x3 x 3x3, schoolbook rank is 18. 
        # Can we do 2x3 x 3x3 in less than 18?
        # Known: rank of 2x3 x 3x3 is at least 15 (from lower bounds)
        # Let's try rank 15 for each block if possible
        
        # Actually, let's just return the schoolbook for now and rely on evolution
        # to find improvements. The key is to have a valid baseline.
        pass
    
    return best
# EVOLVE-BLOCK-END
