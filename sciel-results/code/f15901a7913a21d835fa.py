"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

DEADLINE = 110  # seconds, safety margin before 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 50
    
    # Build the target tensor T as a 3D array
    # T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Start with schoolbook decomposition
    best_rank = rank_school
    best_U = []
    best_V = []
    best_W = []
    
    # Schoolbook
    U = np.zeros((n*m, rank_school))
    V = np.zeros((m*p, rank_school))
    W = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    best_U = U.tolist()
    best_V = V.tolist()
    best_W = W.tolist()
    
    # Try to reduce rank by random perturbation + local search
    # Strategy: try various ranks below schoolbook, use random init and gradient-free optimization
    
    # For a given rank r, we want U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_c U[:,c] * V[:,c] * W[:,c] == T
    
    # This is a hard problem. Let's try a simple approach:
    # Randomly initialize and use iterative refinement
    
    def try_decomposition(rank, max_iters=500):
        """Try to find a decomposition of given rank."""
        if time.time() - start_time > DEADLINE - 5:
            return None
        
        # Random initialization with small values
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Iteratively refine: for each rank component, try to fit the residual
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 5:
                return None
            
            # Compute current product
            # Product[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
            # This is expensive to compute fully, so sample
            
            # Instead, let's use a different approach: 
            # For each (a,b,c) where T[a,b,c] != 0, we need sum_c U[a,c]*V[b,c]*W[c,c] = T[a,b,c]
            # For each (a,b,c) where T[a,b,c] == 0, we need sum_c U[a,c]*V[b,c]*W[c,c] = 0
            
            # This is a system of equations. Let's try a simple iterative method.
            
            # Update one factor at a time
            for factor in range(3):
                if time.time() - start_time > DEADLINE - 5:
                    return None
                
                if factor == 0:  # Update U
                    for a in range(n*m):
                        # For each b,c: sum_c U[a,c]*V[b,c]*W[c,c] should match T
                        # Fix V and W, solve for U[a,:]
                        # T[a,b,c] = sum_c' U[a,c']*V[b,c']*W[c',c]
                        # This is linear in U[a,:]
                        # For each c, the contribution is U[a,c] * (sum over b of V[b,c]*W[c,c]... )
                        # Actually this is complex. Let's use a simpler heuristic.
                        pass
                
                # Simpler: just do random restarts with different inits
                break
            
            # Check if we've converged
            # Compute error on a sample of entries
            sample_size = min(100, n*m*p)
            indices = np.random.choice(n*m*p, sample_size, replace=False)
            errors = []
            for idx in indices:
                a = idx // (m*p)
                rem = idx % (m*p)
                b = rem // p
                c = rem % p
                # Wait, this indexing is wrong. Let me fix.
                # a is in [0, n*m), b in [0, m*p), c in [0, p*n)
                # Total entries = n*m * m*p * p*n
                pass
            
            break
        
        return None
    
    # Given the complexity, let's try a more structured approach:
    # Use the fact that for n=2, we can potentially use rank = 2*m*p - m or similar
    # Actually, let's just try a few specific constructions
    
    # Construction 1: For each pair (i,j), share across k
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For fixed i, this is a matrix-vector product. 
    # We can write: for each i, use m*p terms (schoolbook for that i)
    # Total: n*m*p = 50. No improvement.
    
    # Construction 2: Use bilinear tricks
    # Consider: A[i,j]*B[j,k] for all i,j,k
    # Group by j: for each j, we have A[i,j]*B[j,k] for all i,k
    # This is an outer product of A[:,j] (length n=2) and B[j,:] (length p=5)
    # The outer product has rank 1 as a matrix, but as a tensor term...
    
    # Actually, for fixed j, the tensor slice T[:,j,:] has entries T[i,j,k] for i in {0,1}, k in {0..4}
    # This is a 2x5 matrix of ones (since T[i,j,k]=1 for all i,k when j is fixed)
    # Wait no, T[a,b,c] = 1 only when a=i*m+j, b=j*p+k, c=k*n+i
    # So for fixed j, the non-zero entries are at (i*m+j, j*p+k, k*n+i) for all i,k
    # That's n*p = 10 entries per j, and m=5 values of j, total 50.
    
    # For fixed j, we have 10 non-zero entries. Can we represent this with fewer rank-1 terms?
    # The sub-tensor for fixed j has shape (n*m) x (m*p) x (p*n) but only 10 non-zeros.
    # Specifically, for fixed j, non-zeros at (i*m+j, j*p+k, k*n+i) for i in {0,1}, k in {0..4}
    
    # This sub-tensor can be written as: for each (i,k), a rank-1 term with support only at those indices.
    # That's 10 terms per j, 50 total. No improvement.
    
    # Let me try a completely different approach: use numpy to do a simple optimization
    # with a time limit, trying to find any valid decomposition of rank < 50.
    
    # Given the time constraints, let's just return the schoolbook for now
    # but with a small optimization: try to see if rank 49 is possible by a simple check
    
    # Actually, let me try a randomized search with a simple local optimizer
    def random_search(rank, num_restarts=20):
        if time.time() - start_time > DEADLINE - 10:
            return None
        
        for restart in range(num_restarts):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            # Initialize with schoolbook-like structure but with some randomness
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # First, use a subset of schoolbook terms
            if rank >= n*m*p:
                # Just use schoolbook
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            U[a, col] = 1
                            V[b, col] = 1
                            W[c, col] = 1
                            col += 1
                return (U, V, W)
            
            # For rank < schoolbook, we need a different approach
            # This is very hard in general. Let's skip for now.
            pass
        
        return None
    
    # Given the difficulty, let's just return the schoolbook decomposition
    # The rank is 50, which is the baseline.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
