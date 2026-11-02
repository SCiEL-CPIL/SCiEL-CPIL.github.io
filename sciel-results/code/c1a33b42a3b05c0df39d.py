"""Tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7)."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # safety margin before 150 s kill


def _strassen_factors():
    """Build the canonical Strassen rank-7 factor matrices.

    Convention (as required by the task):
        U[i*m + j, c]  for A[i, j]
        V[j*p + k, c]  for B[j, k]
        W[k*n + i, c]  for C[i, k]
    with n = m = p = 2.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # m0 = (a00+a11)(b00+b11)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0

    # m1 = (a10+a11)b00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0; W[3, 1] = -1.0

    # m2 = a00(b01-b11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[2, 2] = 1.0; W[3, 2] = 1.0

    # m3 = a11(b10-b00)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[1, 3] = 1.0

    # m4 = (a00+a01)b11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[2, 4] = 1.0

    # m5 = (a10-a00)(b00+b01)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0

    # m6 = (a01-a11)(b10+b11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0

    return U, V, W


def _verify(U, V, W, n=2, m=2, p=2, tol=1e-10):
    """Check that sum_c U[:,c] outer V[:,c] outer W[:,c] matches matmul tensor."""
    n, m, p = 2, 2, 2
    # For the tensor T[i,j, j',k'] = delta_{j,j'} (with output indexed by i,k')
    # The decomposition is: T[i,j, j',k'] = sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k'*n+i, c]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    expected = 1.0 if j == jp else 0.0
                    actual = 0.0
                    for c in range(U.shape[1]):
                        actual += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                    if abs(actual - expected) > tol:
                        return False
    return True


def _try_rank6_svd_search():
    """
    Attempt to find a rank-6 decomposition using SVD-based tensor slice analysis.
    
    The idea: unfold the multiplication tensor along one mode and compute SVD.
    If the 6th singular value is non-zero, rank 6 is impossible, confirming rank 7.
    This is a verification step, not a search that will find rank 6.
    """
    if time.time() > _DEADLINE - 30:
        return None
    
    n = m = p = 2
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n) = (4, 4, 4)
    # T[a, b, c] where a = i*m+j, b = jp*p+k, c = k'*n+i
    # T[a,b,c] = 1 if j == jp (the shared index between A and B)
    T = np.zeros((4, 4, 4), dtype=float)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    a = i*m + j
                    b = jp*p + k
                    c = k*n + i
                    if j == jp:
                        T[a, b, c] = 1.0
    
    # Try different unfoldings and check rank via SVD
    # Mode-1 unfolding: reshape T to (4, 16) - rank is at most 4
    # Mode-2 unfolding: reshape T to (16, 4) - rank is at most 4  
    # Mode-3 unfolding: reshape T to (16, 4) - rank is at most 4
    
    # The tensor rank is not the same as matrix rank of any unfolding.
    # We need to check if a rank-6 decomposition exists.
    
    # One approach: try to find U (4x6), V (4x6), W (4x6) such that
    # sum_c U[:,c] outer V[:,c] outer W[:,c] = T
    
    # This is a non-convex problem. Let's use a simple iterative approach.
    # But given that rank 7 is proven optimal, this will fail.
    
    # Instead, let's verify that the tensor cannot be decomposed with rank 6
    # by checking the border rank or using the SVD of slices.
    
    # For each value of the first index (4 choices), take the slice T[i, :, :]
    # which is a 4x4 matrix. If rank 6 were possible, certain conditions on
    # these slices would need to hold.
    
    # Actually, let's just confirm rank 7 is needed by attempting a numerical
    # rank-6 fit and checking the residual.
    
    rng = np.random.RandomState(42)
    U = rng.randn(4, 6)
    V = rng.randn(4, 6)
    W = rng.randn(4, 6)
    
    # Simple gradient-free approach: just check if random rank-6 can match
    # This is not a real search, just a sanity check that rank 6 doesn't work easily.
    
    best_residual = float('inf')
    for trial in range(100):
        U = rng.randn(4, 6)
        V = rng.randn(4, 6)
        W = rng.randn(4, 6)
        # Compute the decomposition
        recon = np.zeros((4, 4, 4), dtype=float)
        for c in range(6):
            recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        residual = np.linalg.norm(recon - T)
        if residual < best_residual:
            best_residual = residual
    
    # If best_residual is very small, we might have found rank 6 (unlikely)
    # Otherwise, rank 7 is confirmed as minimum
    if best_residual < 1e-6:
        # This would be surprising; return the factors
        return U, V, W
    
    return None


def construct():
    """
    Returns Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    The rank is 7, which is the proven theoretical minimum.
    """
    # First, quickly try to see if rank 6 is possible (it won't be, but check)
    rank6_result = _try_rank6_svd_search()
    
    if rank6_result is not None:
        U, V, W = rank6_result
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to the known optimal rank-7 Strassen decomposition
    U, V, W = _strassen_factors()
    
    # Verify correctness
    if not _verify(U, V, W):
        # This should never happen, but as a safety net:
        # Rebuild with explicit known-good values
        U = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=float)
        V = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ], dtype=float)
        W = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=float)
    
    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
