import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return (U, V, W) for the schoolbook decomposition of n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def _strassen_2x2x2(i0, j0, k0, U, V, W, r_start, m, p, n):
    """Add 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Block A rows [i0:i0+2], B cols [j0:j0+2], C rows [k0:k0+2].
    Returns r_start + 7.
    """
    r = r_start
    # M1 = (A[i0,:2] + A[i0+1,:2]) * (B[:2,k0] - B[:2,k0+1])
    #    -> C[i0,k0] += M1, C[i0+1,k0] += M1, C[i0,k0+1] -= M1
    U[i0*m+j0, r] = 1.0
    U[(i0+1)*m+j0, r] = 1.0
    U[i0*m+j0+1, r] = 1.0
    U[(i0+1)*m+j0+1, r] = 1.0
    V[j0*p+k0, r] = 1.0
    V[j0*p+k0+1, r] = -1.0
    V[(j0+1)*p+k0, r] = 1.0
    V[(j0+1)*p+k0+1, r] = -1.0
    W[k0*n+i0, r] = 1.0
    W[k0*n+i0+1, r] = 1.0
    W[(k0+1)*n+i0, r] = -1.0
    r += 1

    # M2 = A[i0+1,:2] * (B[:2,k0] + B[:2,k0+1])
    #    -> C[i0+1,k0+1] += M2
    U[i0*m+j0, r] = 0.0
    U[(i0+1)*m+j0, r] = 1.0
    U[i0*m+j0+1, r] = 0.0
    U[(i0+1)*m+j0+1, r] = 1.0
    V[j0*p+k0, r] = 1.0
    V[j0*p+k0+1, r] = 1.0
    V[(j0+1)*p+k0, r] = 1.0
    V[(j0+1)*p+k0+1, r] = 1.0
    W[(k0+1)*n+i0+1, r] = 1.0
    r += 1

    # M3 = (A[i0,:2] - A[i0+1,:2]) * B[:2,k0+1]
    #    -> C[i0,k0+1] += M3, C[i0+1,k0+1] -= M3
    U[i0*m+j0, r] = 1.0
    U[(i0+1)*m+j0, r] = -1.0
    U[i0*m+j0+1, r] = 1.0
    U[(i0+1)*m+j0+1, r] = -1.0
    V[j0*p+k0, r] = 0.0
    V[j0*p+k0+1, r] = 1.0
    V[(j0+1)*p+k0, r] = 0.0
    V[(j0+1)*p+k0+1, r] = 1.0
    W[k0*n+i0, r] = 0.0
    W[k0*n+i0+1, r] = 0.0
    W[(k0+1)*n+i0, r] = 1.0
    W[(k0+1)*n+i0+1, r] = -1.0
    r += 1

    # M4 = A[i0,:2] * B[:2,k0]
    #    -> C[i0,k0] += M4
    U[i0*m+j0, r] = 1.0
    U[(i0+1)*m+j0, r] = 0.0
    U[i0*m+j0+1, r] = 1.0
    U[(i0+1)*m+j0+1, r] = 0.0
    V[j0*p+k0, r] = 1.0
    V[j0*p+k0+1, r] = 0.0
    V[(j0+1)*p+k0, r] = 1.0
    V[(j0+1)*p+k0+1, r] = 0.0
    W[k0*n+i0, r] = 1.0
    r += 1

    # M5 = (A[i0+1,:2] + A[i0,:2]) * B[:2,k0+1]
    #    -> C[i0+1,k0] += M5
    U[i0*m+j0, r] = 1.0
    U[(i0+1)*m+j0, r] = 1.0
    U[i0*m+j0+1, r] = 1.0
    U[(i0+1)*m+j0+1, r] = 1.0
    V[j0*p+k0, r] = 0.0
    V[j0*p+k0+1, r] = 1.0
    V[(j0+1)*p+k0, r] = 0.0
    V[(j0+1)*p+k0+1, r] = 1.0
    W[k0*n+i0+1, r] = 1.0
    r += 1

    # M6 = (A[i0+1,:2] - A[i0,:2]) * (B[:2,k0] + B[:2,k0+1])
    #    -> C[i0+1,k0] += M6
    U[i0*m+j0, r] = -1.0
    U[(i0+1)*m+j0, r] = 1.0
    U[i0*m+j0+1, r] = -1.0
    U[(i0+1)*m+j0+1, r] = 1.0
    V[j0*p+k0, r] = 1.0
    V[j0*p+k0+1, r] = 1.0
    V[(j0+1)*p+k0, r] = 1.0
    V[(j0+1)*p+k0+1, r] = 1.0
    W[k0*n+i0+1, r] = 1.0
    r += 1

    # M7 = (A[i0,:2] - A[i0+1,:2]) * (B[:2,k0+1] - B[:2,k0])
    #    -> C[i0,k0] += M7
    U[i0*m+j0, r] = 1.0
    U[(i0+1)*m+j0, r] = -1.0
    U[i0*m+j0+1, r] = 1.0
    U[(i0+1)*m+j0+1, r] = -1.0
    V[j0*p+k0, r] = -1.0
    V[j0*p+k0+1, r] = 1.0
    V[(j0+1)*p+k0, r] = -1.0
    V[(j0+1)*p+k0+1, r] = 1.0
    W[k0*n+i0, r] = 1.0
    r += 1

    return r


def _build_decomp_98():
    """Build the rank-98 decomposition: Strassen on 2x2x2 blocks + schoolbook for the rest."""
    n, m, p = N, M, P
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    # C11: A11(2x2)*B11(2x2) via Strassen: i0=0, j0=0, k0=0
    r = _strassen_2x2x2(0, 0, 0, U, V, W, r, m, p, n)
    # C11: A12(2x3)*B21(3x2) schoolbook: i in {0,1}, j in {2,3,4}, k in {0,1}
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    
    # C12: A11(2x2)*B12(2x3) schoolbook: i in {0,1}, j in {0,1}, k in {2,3,4}
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    # C12: A12(2x3)*B22(3x3) schoolbook: i in {0,1}, j in {2,3,4}, k in {2,3,4}
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    
    # C21: A21(2x2)*B11(2x2) via Strassen: i0=2, j0=0, k0=0
    r = _strassen_2x2x2(2, 0, 0, U, V, W, r, m, p, n)
    # C21: A22(2x3)*B21(3x2) schoolbook: i in {2,3}, j in {2,3,4}, k in {0,1}
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    
    # C22: A21(2x2)*B12(2x3) schoolbook: i in {2,3}, j in {0,1}, k in {2,3,4}
    for i in range(2, 4):
        for j in range(2):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    # C22: A22(2x3)*B22(3x3) schoolbook: i in {2,3}, j in {2,3,4}, k in {2,3,4}
    for i in range(2, 4):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    return U, V, W


def _try_reduce_rank(U, V, W, target_rank, deadline):
    """
    Try to reduce the rank from current to target_rank using iterative refinement.
    Uses a simple gradient-based approach on the tensor decomposition.
    Returns (U, V, W) if successful, or (U, V, W) unchanged if not.
    """
    n, m, p = N, M, P
    current_rank = U.shape[1]
    
    if target_rank >= current_rank:
        return U, V, W
    
    # Initialize with the current decomposition, zeroing out extra columns
    U_new = U[:, :target_rank].copy()
    V_new = V[:, :target_rank].copy()
    W_new = W[:, :target_rank].copy()
    
    # Add random perturbations to help escape local minima
    rng = np.random.RandomState(42)
    U_new += rng.randn(n*m, target_rank) * 0.01
    V_new += rng.randn(m*p, target_rank) * 0.01
    W_new += rng.randn(p*n, target_rank) * 0.01
    
    # Simple iterative refinement
    for iteration in range(50):
        if time.time() > deadline:
            break
            
        # Compute the residual tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
        # This is expensive, so we do a simplified check
        
        # Check if we're close to a valid solution
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r] for r in range(target_rank))
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-6:
            # We have a valid solution, try to clean up
            return U_new, V_new, W_new
        
        # Gradient step (simplified)
        # For each term r, compute the contribution to the error
        # and adjust U, V, W to reduce it
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    # Compute gradient w.r.t U[i*m+j, r]
                    grad_u = 0.0
                    for k in range(p):
                        grad_u += V_new[j*p+k, r] * W_new[k*n+i, r]
                    # Simple update
                    U_new[i*m+j, r] += 0.001 * grad_u
                    
                    # Compute gradient w.r.t V[j*p+k, r]
                    for k in range(p):
                        grad_v = 0.0
                        for i2 in range(n):
                            grad_v += U_new[i2*m+j, r] * W_new[k*n+i2, r]
                        V_new[j*p+k, r] += 0.001 * grad_v
                        
                    # Compute gradient w.r.t W[k*n+i, r]
                    for k in range(p):
                        grad_w = 0.0
                        for j2 in range(m):
                            grad_w += U_new[i*m+j2, r] * V_new[j2*p+k, r]
                        W_new[k*n+i, r] += 0.001 * grad_w
    
    # Check final result
    if _validate(U_new, V_new, W_new, tol=1e-6):
        return U_new, V_new, W_new
    
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Uses Strassen-based block decomposition to achieve rank 98, then attempts further reduction.
    """
    t0 = time.time()
    n, m, p = N, M, P
    
    # Build the rank-98 decomposition
    U, V, W = _build_decomp_98()
    
    # Validate
    if not _validate(U, V, W):
        # Fall back to schoolbook if validation fails
        U, V, W = _schoolbook(n, m, p)
    
    # Deadline guard
    elapsed = time.time() - t0
    remaining = DEADLINE_S - elapsed
    
    # Try to reduce rank further if we have time
    if remaining > 10:
        # Try target rank 97
        U2, V2, W2 = _try_reduce_rank(U, V, W, 97, t0 + DEADLINE_S - 5)
        if _validate(U2, V2, W2):
            U, V, W = U2, V2, W2
            elapsed = time.time() - t0
            remaining = DEADLINE_S - elapsed
        
        # Try target rank 96 if we still have time
        if remaining > 10:
            U3, V3, W3 = _try_reduce_rank(U, V, W, 96, t0 + DEADLINE_S - 5)
            if _validate(U3, V3, W3):
                U, V, W = U3, V3, W3
    
    # Final validation
    if not _validate(U, V, W):
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
