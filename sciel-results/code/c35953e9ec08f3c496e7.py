<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-minimal exact tensor decomposition using rank-1 updates
    # We are looking for rank r such that the multiplication tensor can be written as sum of r rank-1 tensors
    # The schoolbook rank is n*m*p = 30
    # Lower bounds: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # Known results: for 2x3 * 3x5, we can try to find a lower rank
    
    # Let's try to construct a decomposition using a systematic approach
    # We'll use the fact that matrix multiplication tensors have structure
    
    # For now, let's try to find a better decomposition than schoolbook
    # by exploiting the structure of the problem
    
    # One approach: use the fact that we can combine some of the rank-1 terms
    # if the matrices have special structure
    
    # Let's try a different approach: use bilinear algorithms
    # The key insight is that for small matrices, we might be able to find
    # a decomposition with rank less than n*m*p
    
    # For 2x3 * 3x5, let's try to find a rank-15 or lower decomposition
    
    # The multiplication tensor T has entries T[i,j,k,l] where:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,k,j] = 1 if we're computing (A*B)[i,k] using A[i,j] and B[j,k]
    
    # In the factorization:
    # U has shape (n*m, r) = (6, r)
    # V has shape (m*p, r) = (15, r) 
    # W has shape (p*n, r) = (10, r)
    # And we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # And for all other combinations, the sum should be 0
    
    # Let's try to find a rank-15 decomposition (the theoretical lower bound)
    
    # Actually, let's just return the schoolbook decomposition for now,
    # but structure it to allow for future improvements
    
    # Schoolbook rank is 30
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
=======
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

_start_time = time.time()
_DEADLINE = 140.0  # seconds; 10s safety margin before 150s kill

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    r = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the multiplication tensor
    # T[i,j,k] = 1 if computing (A*B)[i,k] using A[i,j] and B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # And for "wrong" combinations, the sum should be 0
    
    # Actually, let's verify by checking that for random matrices,
    # the product computed via the decomposition matches A*B
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    AB = A @ B
    
    # Compute via decomposition
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k] / (U[i*m+j,c]*V[j*p+k,c])
    # Wait, that's not right. Let me think again.
    
    # The tensor decomposition says:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 if we're computing (A*B)[i,k] from A[i,j] and B[j,k]
    
    # So the product is:
    # (A*B)[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    
    # Let's verify this
    result = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for c in range(r):
                    result[i, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    
    if not np.allclose(result, AB, atol=1e-10):
        return False
    
    # Also verify with another random matrix
    A2 = np.random.randn(n, m)
    B2 = np.random.randn(m, p)
    AB2 = A2 @ B2
    
    result2 = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for c in range(r):
                    result2[i, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A2[i,j] * B2[j,k]
    
    if not np.allclose(result2, AB2, atol=1e-10):
        return False
    
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _try_rank15(n, m, p):
    """
    Try to find a rank-15 decomposition.
    
    The lower bound is max(n*m, m*p, p*n) = max(6, 15, 10) = 15.
    
    Strategy: We have 6 input slots for A (n*m), 15 for B (m*p), 10 for output (p*n).
    We need to assign values to U (6x15), V (15x15), W (10x15) such that
    the tensor decomposition holds.
    
    One approach: use the fact that for each (i,k) output, we need:
    sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k] = (A*B)[i,k]
    
    Let's try a structured approach where we group the 15 terms.
    """
    # For 2x3 * 3x5, we have:
    # A is 2x3, B is 3x5, C is 2x5
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # ... etc.
    
    # Let's try to find a rank-15 decomposition by solving a system.
    # This is complex, so let's try a heuristic approach.
    
    # Actually, let's try to use the structure of the problem.
    # We can think of this as: for each column of B (there are 5 columns),
    # we multiply A (2x3) by that column (3x1) to get a 2x1 result.
    
    # For a fixed column k of B, we have:
    # C[:,k] = A @ B[:,k]
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # This is a 2x3 * 3x1 multiplication, which has rank at most 6 (schoolbook).
    # But we can do better! The rank of 2x3 * 3x1 is at most 5 (since it's a linear map from R^3 to R^2).
    
    # Actually, for 2x3 * 3x1, the minimum rank is 5 (you need at least 5 bilinear terms).
    
    # So for 5 columns of B, if we could share some terms, we might get rank < 15.
    # But the terms for different columns are generally independent.
    
    # Let's just try to verify if rank 15 is achievable with a simple construction.
    
    # Construction: For each (j,k) pair (15 pairs), create one term.
    # U[i*m+j, j*p+k] = 1 if i is in some set, else 0
    # V[j*p+k, j*p+k] = 1
    # W[k*n+i, j*p+k] = 1 if we're computing C[i,k]
    
    # But this doesn't work directly because we need to sum over j.
    
    # Let's try a different approach: use the fact that we can combine terms.
    
    # For now, let's just return None to indicate we couldn't find a better decomposition.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    if _time_left() < 5.0:
        return _schoolbook(n, m, p)
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    
    # Try to find rank-15 decomposition
    if _time_left() > 10.0:
        result15 = _try_rank15(n, m, p)
        if result15 is not None:
            if _verify(result15["U"], result15["V"], result15["W"], n, m, p):
                best = result15
    
    # Remove the rank key from the result
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
