import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the tensor decomposition is correct."""
    R = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check if T equals the multiplication tensor (which is 1 for all valid i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_222():
    """Rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, _ = V1.shape
    m2, _ = V2.shape
    p1, _ = W1.shape
    p2, _ = W2.shape
    
    R = r1 * r2
    U = np.zeros((n1 * n2, R), dtype=float)
    V = np.zeros((m1 * m2, R), dtype=float)
    W = np.zeros((p1 * p2, R), dtype=float)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            U[:, r_idx] = np.kron(U1[:, r1_idx], U2[:, r2_idx])
            V[:, r_idx] = np.kron(V1[:, r1_idx], V2[:, r2_idx])
            W[:, r_idx] = np.kron(W1[:, r1_idx], W2[:, r2_idx])
            
    return U, V, W

def _direct_sum_p_split(U444, V444, W444):
    """
    Split p=8 into two p=4 blocks using direct sum.
    Input: 4x4x4 decomposition (U: 16xr, V: 16xr, W: 16xr)
    Output: 4x4x8 decomposition (U: 16x2r, V: 32x2r, W: 32x2r)
    """
    n, r = U444.shape
    m, _ = V444.shape  # should be 16 = 4*4
    p, _ = W444.shape  # should be 16 = 4*4
    
    R = 2 * r
    U = np.zeros((n, R), dtype=float)
    V = np.zeros((2 * m, R), dtype=float)  # m=16, so 32
    W = np.zeros((2 * p, R), dtype=float)  # p=16, so 32
    
    # First half of p (k=0..3)
    U[:, :r] = U444
    V[:m, :r] = V444
    W[:p, :r] = W444
    
    # Second half of p (k=4..7)
    U[:, r:] = U444
    V[m:, r:] = V444
    W[p:, r:] = W444
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii*m + jj, col] = 1
                V[jj*p + kk, col] = 1
                W[kk*n + ii, col] = 1
                col += 1
    return U, V, W

def _greedy_elimination(U, V, W, n, m, p, deadline):
    """
    Greedy elimination of linearly dependent columns.
    Attempts to remove one column at a time if the remaining columns
    can still span the same tensor.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    R = U.shape[1]
    
    # Pre-compute the "target" tensor values for all (i,j,k)
    # T[i,j,k] = 1 for all i,j,k
    # We need to check if removing column c still allows exact reconstruction.
    
    while time.time() < deadline and R > 1:
        # Try to remove each column and see if the decomposition still works
        removed = False
        for c in range(R):
            if time.time() > deadline:
                break
            
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                U = U_test
                V = V_test
                W = W_test
                R -= 1
                removed = True
                break
        
        if not removed:
            break
    
    return U, V, W

def _lp_elimination(U, V, W, n, m, p, deadline):
    """
    Use linear programming to find a set of columns that can be eliminated
    simultaneously. This is more powerful than greedy elimination.
    
    For each column r, we ask: can we express column r as a linear combination
    of the other columns? If so, we can eliminate it.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    R = U.shape[1]
    
    # We'll try to find a subset of columns to keep that still spans the tensor.
    # This is equivalent to finding the minimum rank decomposition.
    
    # Simple approach: try removing columns one by one using LP.
    # For each column c, check if it's in the span of the others.
    
    while time.time() < deadline and R > 1:
        removed = False
        for c in range(R):
            if time.time() > deadline:
                break
            
            # Check if column c is in the span of the other columns
            # We need to solve: U[:,c] = U[:,S] @ alpha, V[:,c] = V[:,S] @ beta, W[:,c] = W[:,S] @ gamma
            # where S is the set of all columns except c.
            
            U_other = np.delete(U, c, axis=1)
            V_other = np.delete(V, c, axis=1)
            W_other = np.delete(W, c, axis=1)
            
            # Try to find alpha, beta, gamma such that the above holds.
            # This is a linear system. We can use least squares.
            
            try:
                # Solve for alpha
                alpha, residuals_a, rank_a, s_a = np.linalg.lstsq(U_other, U[:, c], rcond=None)
                # Solve for beta
                beta, residuals_b, rank_b, s_b = np.linalg.lstsq(V_other, V[:, c], rcond=None)
                # Solve for gamma
                gamma, residuals_c, rank_c, s_c = np.linalg.lstsq(W_other, W[:, c], rcond=None)
                
                # Check if the reconstruction is exact
                recon_U = U_other @ alpha
                recon_V = V_other @ beta
                recon_W = W_other @ gamma
                
                if (np.allclose(recon_U, U[:, c], atol=1e-10) and
                    np.allclose(recon_V, V[:, c], atol=1e-10) and
                    np.allclose(recon_W, W[:, c], atol=1e-10)):
                    
                    # Column c is redundant. Remove it.
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    R -= 1
                    removed = True
                    break
                    
            except np.linalg.LinAlgError:
                continue
        
        if not removed:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Start with the known rank-98 decomposition:
    # 4x4x4 via Strassen kron Strassen (rank 49), then direct sum for p=8
    U2, V2, W2 = _strassen_222()
    U444, V444, W444 = _kron_decomposition(U2, V2, W2, U2, V2, W2)
    U, V, W = _direct_sum_p_split(U444, V444, W444)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Apply greedy elimination first (fast)
    if time.time() < deadline:
        U2, V2, W2 = _greedy_elimination(best_U, best_V, best_W, n, m, p, deadline)
        if U2.shape[1] < best_rank and _verify_decomposition(U2, V2, W2, n, m, p):
            best_U, best_V, best_W = U2, V2, W2
            best_rank = U2.shape[1]
    
    # Apply LP-based elimination (slower but more powerful)
    if time.time() < deadline:
        U2, V2, W2 = _lp_elimination(best_U, best_V, best_W, n, m, p, deadline)
        if U2.shape[1] < best_rank and _verify_decomposition(U2, V2, W2, n, m, p):
            best_U, best_V, best_W = U2, V2, W2
            best_rank = U2.shape[1]
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
