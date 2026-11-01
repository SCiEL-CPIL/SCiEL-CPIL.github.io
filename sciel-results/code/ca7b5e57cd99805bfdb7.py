"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return U, V, W as lists for the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Returns True if valid (within numerical tolerance), False otherwise.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer (i.e., 2*x is an integer)."""
    return abs(2 * x - round(2 * x)) < 1e-9

def _verify_entries(U, V, W):
    """Check all entries are integers or half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _try_structured_rank(n, m, p, rank, deadline):
    """
    Try to find a rank-`rank` decomposition using a structured approach.
    For n=2, we can exploit the structure:
    C[0,k] = sum_j A[0,j] B[j,k]
    C[1,k] = sum_j A[1,j] B[j,k]
    
    We'll use a parameterized family and solve for coefficients.
    """
    if time.time() > deadline:
        return None
    
    # Strategy: Use the fact that for n=2, we can write
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    
    # A common technique: introduce auxiliary variables to combine terms.
    # For each pair (j, k), we need to compute A[0,j]*B[j,k] and A[1,j]*B[j,k].
    # We can use the identity: x*y = ((x+y)^2 - x^2 - y^2)/4
    # But this increases rank.
    
    # Alternative: use bilinear forms with shared factors.
    # Let's try a random initialization approach with alternating least squares.
    
    import random
    random.seed(42)
    
    # Initialize with random values
    U = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(p * n)]
    
    # Alternating least squares: fix two, solve for the third.
    # This is complex for exact solutions. Let's try a simpler approach.
    
    # For now, this is a placeholder. The main improvement will come from
    # trying known low-rank constructions.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if _verify_decomposition(U, V, W, n, m, p) and _verify_entries(U, V, W):
        # Schoolbook is valid
        pass
    
    # For [2, 5, 6], the schoolbook rank is 60.
    # Let's try to find a better decomposition.
    
    # Known results: For [2, m, p], the rank can sometimes be less than 2*m*p.
    # One approach: use the fact that we can share some terms between the two rows.
    
    # Let's try a construction based on combining pairs.
    # For each j in [0,5) and k in [0,6), we have two products:
    # A[0,j]*B[j,k] and A[1,j]*B[j,k]
    
    # We can write:
    # A[0,j]*B[j,k] + A[1,j]*B[j,k] = (A[0,j]+A[1,j])*B[j,k]
    # A[0,j]*B[j,k] - A[1,j]*B[j,k] = (A[0,j]-A[1,j])*B[j,k]
    
    # So:
    # A[0,j]*B[j,k] = 0.5*((A[0,j]+A[1,j]) + (A[0,j]-A[1,j])) * B[j,k]
    # A[1,j]*B[j,k] = 0.5*((A[0,j]+A[1,j]) - (A[0,j]-A[1,j])) * B[j,k]
    
    # This doesn't reduce rank directly, but suggests a structure.
    
    # Let's try a different approach: use the Winograd scheme or similar.
    # For now, we'll stick with schoolbook since finding a provably better
    # exact decomposition with half-integer entries is non-trivial.
    
    # However, let's try one specific construction:
    # Group terms by j. For each j, we have 6 terms (one for each k).
    # We can potentially combine some using bilinear identities.
    
    # Given the complexity and time constraints, return schoolbook.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
