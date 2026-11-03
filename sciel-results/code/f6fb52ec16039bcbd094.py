"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The multiplication tensor: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually the tensor for matmul is: for each (i,j,k), the coefficient is 1
    # We need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta(i,j,k) pattern
    # The matmul tensor T has T[i,j,k] = 1 for all valid (i,j,k) in the multiplication pattern
    # Specifically: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T_{i,j,k} = 1 (it's the all-ones pattern in the appropriate sense)
    # Wait, let me reconsider. The tensor is:
    # T[i, j, k] = 1 for the multiplication: output[i,k] gets contribution A[i,j]*B[j,k]
    # So for each triple (i,j,k), T[i,j,k] = 1
    # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Actually no. The tensor decomposition for matrix multiplication is:
    # M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where M[i,j,k] = 1 for all i,j,k (since every A[i,j] contributes to every C[i,k] via B[j,k])
    # Wait, that's not right either. Let me think again.
    
    # The matrix multiplication tensor T is defined as:
    # T[i, j, k] = 1 if the (i,j) element of A and (j,k) element of B contribute to (i,k) element of C
    # In standard matmul: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # That means T is the all-ones tensor of shape (n, m, p)? No...
    
    # Actually, the tensor for bilinear multiplication is:
    # For each output position (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k (it's rank-1 in a trivial sense if all ones)
    # No wait. The tensor is: T[i,j,k] represents the coefficient such that
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _try_structured_decompositions(n, m, p, deadline):
    """Try various structured decompositions to find lower rank."""
    best = None
    best_rank = n * m * p
    
    # Try rank 1 to best_rank-1
    # This is expensive, so we'll use a smarter approach
    
    # Approach 1: Use the structure that T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of n*m*p equations in rank variables (3*rank unknowns)
    # We can try to find sparse solutions
    
    # Approach 2: Try decompositions based on known algebraic identities
    # For example, if we can write the all-ones tensor as a sum of fewer rank-1 tensors
    
    # The all-ones tensor of shape (a,b,c) has rank min(a,b,c) if we allow arbitrary entries
    # But here the indices are coupled: (i*m+j, j*p+k, k*n+i)
    
    # Let me try a direct approach: assume U, V, W have entries in {0, 1, -1} or small integers
    # and search for low-rank solutions
    
    # For now, let's try a numerical approach using least squares or iterative methods
    
    # Try ranks from low to high
    for rank in range(1, best_rank):
        if time.time() > deadline:
            break
        # Try to find a decomposition of this rank
        # Use a simple iterative approach or random initialization
        
        # Random initialization and verification
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            np.random.seed(rank * 1000 + attempt)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # This is just a random guess; verification will almost certainly fail
            # We need a smarter approach
        
        # Better approach: use the fact that we need exact solutions
        # Try structured patterns
    
    # For now, return the schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    return U, V, W, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    U, V, W, best_rank = _try_structured_decompositions(n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
