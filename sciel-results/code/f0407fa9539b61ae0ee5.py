"""Rewritten program for 3x3x6 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # The matrix multiplication tensor T for (n x m) * (m x p) -> (n x p)
    # T[(i,j), (j',k), (k',i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    #
    # We want to find rank-minimal U, V, W such that:
    # T[(i,j), (j',k), (k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    #
    # U shape: (n*m, R) = (9, R)
    # V shape: (m*p, R) = (18, R)
    # W shape: (p*n, R) = (18, R)
    #
    # Key insight: For each (i, k) pair, we need:
    # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'}
    #
    # This is a linear system for W[k*n+i, :].
    # Let A^{(i,k)} be a 9 x R matrix where A^{(i,k)}[(j,j'), r] = U[i*m+j, r] * V[j'*p+k, r]
    # Then A^{(i,k)} @ w = b where b is the vectorized identity (9 elements: 1 at (j,j) for j=0,1,2)
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        R = U.shape[1]
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for i2 in range(n):
                                expected = 1.0 if (i == i2 and j == jp and k == kp) else 0.0
                                val = 0.0
                                for r in range(R):
                                    val += U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+i2, r]
                                err = abs(val - expected)
                                if err > max_err:
                                    max_err = err
        return max_err < tol, max_err

    # Build schoolbook decomposition as baseline
    def build_schoolbook():
        R = n * m * p  # 54
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    best_rank = n * m * p
    best_U, best_V, best_W = build_schoolbook()

    # Strategy: Try to find low-rank decompositions
    # For each candidate rank R, randomly initialize U and V, then solve for W
    # Check if the solution works exactly (within tolerance)
    
    rng = np.random.RandomState(42)
    
    # Try ranks from small to large
    # The tensor T has a special structure. Let's analyze what's achievable.
    #
    # For n=m=3, p=6: The tensor T[(i,j),(j',k),(k',i')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    #
    # One approach: Use block structure. Split the 6 columns into two groups of 3.
    # For k in {0,1,2} and k in {3,4,5}, we have independent sub-problems of size 3x3x3.
    #
    # The 3x3x3 matrix multiplication tensor has known rank. The Strassen algorithm
    # gives rank 7 for 3x3x3 (actually Strassen is for 2x2x2 giving rank 7, and
    # for 3x3x3 the best known is around 19-23).
    #
    # Actually, let me reconsider. The standard result:
    # - 2x2x2: rank 7 (Strassen)
    # - 3x3x3: best known rank is 19 (Håstad)
    # - 4x4x4: best known rank is around 49
    #
    # For 3x3x6, if we can decompose it as two 3x3x3 blocks, we'd get rank 2*19 = 38.
    # But maybe we can do better by sharing structure between the two blocks.
    #
    # Let's try a direct approach: random search with the linear system method.
    
    def try_rank(R, n_attempts=50):
        """Try to find a valid decomposition of rank R."""
        for attempt in range(n_attempts):
            # Randomly initialize U and V
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            
            # For each (i, k) pair, solve for W[k*n+i, :]
            W = np.zeros((p*n, R))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the 9 x R matrix A
                    # A[(j, jp), r] = U[i*m+j, r] * V[jp*p+k, r]
                    A = np.zeros((m*m, R))
                    b = np.zeros(m*m)
                    for j in range(m):
                        for jp in range(m):
                            A[j*m+jp, :] = U[i*m+j, :] * V[jp*p+k, :]
                            if j == jp:
                                b[j*m+jp] = 1.0
                    
                    # Solve A @ w = b using least squares
                    if R >= m*m:
                        # Overdetermined or square - use least squares
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        # Check if the solution is close to integers or half-integers
                        # For exact decomposition, we need A @ w = b exactly
                        residual = np.linalg.norm(A @ w - b)
                        if residual > 1e-8:
                            valid = False
                            break
                    else:
                        # Underdetermined - check if there exists a solution
                        # Project b onto the column space of A
                        # A @ w = b has a solution iff b is in the column space of A
                        # Check: rank([A|b]) == rank(A)
                        Ab = np.column_stack([A, b])
                        rank_A = np.linalg.matrix_rank(A)
                        rank_Ab = np.linalg.matrix_rank(Ab)
                        if rank_A != rank_Ab:
                            valid = False
                            break
                        # Find one solution
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    
                    W[k*n+i, :] = w
            
            if not valid:
                continue
            
            # Verify the full decomposition
            ok, max_err = verify(U, V, W, tol=1e-7)
            if ok:
                return U, V, W
        
        return None

    # Try various ranks
    # Start from reasonable lower bounds
    # Lower bound: max(n*m, m*p, p*n) / something... actually the lower bound is more complex.
    # For 3x3x3, the lower bound is 19 (Håstad). For 3x3x6, it's at least 19.
    # Let's try ranks from 19 upward.
    
    # But first, let's try a structured approach for 3x3x6.
    # Split into two 3x3x3 blocks.
    # For the first block (k=0,1,2), we need a 3x3x3 decomposition.
    # For the second block (k=3,4,5), we need another 3x3x3 decomposition.
    # If we use the same U for both blocks (which is possible since U depends only on i,j),
    # we can share the U factors.
    
    # Actually, let me think more carefully.
    # U[i*m+j, r] is the same for both k blocks.
    # V[j*p+k, r] for k=0,1,2 is one set, and V[j*p+k, r] for k=3,4,5 is another set.
    # W[k*n+i, r] similarly.
    
    # So we can try:
    # 1. Find a 3x3x3 decomposition (U3, V3a, W3a) for the first block
    # 2. Find a 3x3x3 decomposition (U3b, V3b, W3b) for the second block
    # 3. If U3a and U3b are the same (or can be made the same), combine them.
    
    # But the Strassen algorithm for 3x3x3 might not have a simple shared U structure.
    
    # Let's just try the random search approach with the linear system method.
    
    for R in range(19, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        result = try_rank(R, n_attempts=100)
        if result is not None:
            U, V, W = result
            best_rank = R
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to list format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
