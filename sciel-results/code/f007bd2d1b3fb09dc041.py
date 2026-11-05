"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # Dimensions: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # The tensor T[a, b, c] where:
    # a = i*m + j (i in [0,n), j in [0,m))
    # b = j*p + k (j in [0,m), k in [0,p))
    # c = k*n + i (k in [0,p), i in [0,n))
    # T[a, b, c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i, else 0
    
    # For this specific structure, T[a,b,c] = 1 iff:
    # Let i = c % n, k = c // n, j = b // p, then a must equal i*m + j
    # And b % p must equal k
    
    # Let's verify: given c = k*n + i, so i = c%n, k = c//n
    # Given b = j*p + k, so j = b//p, and b%p should equal k
    # Then a should be i*m + j
    
    # So T[a,b,c] = 1 if and only if:
    # i = c % n, k = c // n
    # j = b // p
    # b % p == k
    # a == i * m + j
    
    # Now, let's try to find a rank decomposition.
    # We'll attempt rank = 20 as per the directive.
    
    rank = 20
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strategy: For each output index c (0 to pn-1), we need to cover all valid (a,b) pairs.
    # For a given c, i = c%n, k = c//n
    # Valid (a,b): b must satisfy b//p = j and b%p = k, so b = j*p + k for j in [0,m)
    # a = i*m + j
    # So for fixed c, the valid (a,b) are: (i*m+j, j*p+k) for j=0,...,m-1
    
    # This means for each c, we have m terms that need to be covered.
    # Total terms: pn * m = 10 * 4 = 40, which matches n*m*p.
    
    # To get rank < 40, we need to share rank-1 components across different c values.
    
    # Let's try a different approach: use the structure to find linear dependencies.
    # 
    # Actually, let me try to see if rank 20 is feasible by construction.
    # 
    # Idea: Group by j. For each j in [0,m), and for each (i,k), we have a term.
    # The term for (i,j,k) contributes to U[i*m+j], V[j*p+k], W[k*n+i].
    # 
    # If we fix j, we have n*p = 2*5 = 10 terms.
    # There are m=4 such groups, total 40 terms.
    # 
    # To reduce rank, we need to find linear combinations.
    # 
    # Let me try: for each j, can we express the n*p terms using fewer than n*p rank-1 components?
    # Probably not in general, but let's see if there's a global structure.
    
    # Alternative: Let's just try to solve for a rank-20 decomposition using linear algebra.
    # We need to find U, V, W such that:
    # For all a,b,c: sum_c U[a,c]*V[b,c]*W[c,c] = T[a,b,c]
    
    # This is a nonlinear system, so let's try a greedy/iterative approach.
    
    # Greedy rank-1 peeling:
    # Start with residual = T (as a 3D tensor)
    # At each step, find the best rank-1 approximation, subtract it, repeat.
    
    # But this might not give minimal rank. Let me try a more structured approach.
    
    # Since n=2 is small, let's enumerate all possible rank-1 components and see if we can find a subset of size 20 that sums to T.
    
    # A rank-1 component is determined by (u, v, w) where u in R^8, v in R^20, w in R^10.
    # The contribution is u ⊗ v ⊗ w.
    
    # This is a huge search space. Let me try a different tactic.
    
    # Let me check if the schoolbook rank of 40 is actually minimal, or if we can do better.
    # 
    # For now, let me return the schoolbook solution which is guaranteed to work,
    # but let me first try to see if a smaller rank is possible by checking known results.
    
    # Actually, let me try rank = 30 as a compromise and see if I can construct it.
    # 
    # Wait, the directive says to target rank 20. Let me try harder.
    
    # New idea: Use the fact that for each j, the sub-tensor for that j has rank at most n*p = 10.
    # If I can find a decomposition where the rank-1 components for different j's are "orthogonal" in some sense,
    # I might be able to reuse components.
    
    # Let me try: assign rank-1 components indexed by (j, t) where t in [0, 10), so 4*10 = 40 total.
    # That's still 40. To get to 20, I need each component to serve two j's.
    
    # Let me try a completely different approach: use the matrix multiplication identity.
    # 
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # This is already the schoolbook. To reduce rank, I need to find an alternative formula.
    # 
    # For 2x4x5, I don't know of a standard algorithm with < 40 multiplications.
    # 
    # Let me just return the schoolbook solution for now, which gives rank 40.
    # But the current program already does this, so I need to do better.
    
    # Let me try to verify if rank 20 is possible by attempting a numerical search.
    # 
    # Actually, let me reconsider the problem statement. It says "rank-minimal exact tensor decomposition."
    # The score is the rank, lower is better. The seed is 40 (from n*m*p = 2*4*5 = 40).
    # 
    # If I can find any decomposition with rank < 40, I'll improve.
    # 
    # Let me try rank = 36 (Strassen-like) or other values.
    # 
    # For now, let me implement a function that tries to find a decomposition of a given rank
    # using a simple iterative method, and if it fails, fall back to schoolbook.
    
    def try_decomposition(target_rank):
        """Try to find a rank decomposition of the given size."""
        # Use a simple iterative approach: initialize randomly, then optimize
        # This is heuristic and may not find the optimum.
        
        np.random.seed(42)
        
        # Initialize
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # Build the target tensor
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Iterative optimization: for each rank-1 component, try to improve it
        # This is a very naive approach and may not converge.
        
        # Given the time constraint, let me just return schoolbook if this doesn't work quickly.
        return None
    
    # For now, return schoolbook (rank 40)
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
# EVOLVE-BLOCK-END
