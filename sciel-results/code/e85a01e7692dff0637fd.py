import time
import numpy as np

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the tensor decomposition is correct."""
    R = U.shape[1]
    # Efficient verification using matrix operations
    # For each rank component r, we need:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 for all i,j,k
    # This is equivalent to checking that the tensor equals the multiplication tensor
    
    # Build the full multiplication tensor and compare
    # T[i,j,k] = 1 if the output (i,k) depends on input (i,j) and (j,k)
    # Actually, T[i,j,k] = 1 always for valid matrix mult indices
    
    # More efficient: for each r, form outer products and check
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all valid i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check a sample of entries for speed, then full check if needed
    n_check = min(n, 2)
    m_check = min(m, 2)  
    p_check = min(p, 2)
    
    for i in range(n_check):
        for j in range(m_check):
            for k in range(p_check):
                s = 0.0
                for r in range(R):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Full verification
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

def _greedy_column_elimination(U, V, W, n, m, p, deadline):
    """
    Try to eliminate columns by checking if a column can be expressed
    as a linear combination of others in all three matrices simultaneously.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        R = U.shape[1]
        if R <= 1:
            break
        
        # Try each column from the end
        for k in range(R - 1, -1, -1):
            if time.time() > deadline:
                return U, V, W
            
            U_rest = np.delete(U, k, axis=1)
            V_rest = np.delete(V, k, axis=1)
            W_rest = np.delete(W, k, axis=1)
            
            try:
                # Build the joint system
                stacked_A = np.vstack([U_rest, V_rest, W_rest])
                stacked_b = np.concatenate([U[:, k], V[:, k], W[:, k]])
                
                coeffs, residuals, rank, _ = np.linalg.lstsq(stacked_A, stacked_b, rcond=None)
                residual = np.linalg.norm(stacked_A @ coeffs - stacked_b)
                
                if residual < 1e-7:
                    U = np.delete(U, k, axis=1)
                    V = np.delete(V, k, axis=1)
                    W = np.delete(W, k, axis=1)
                    changed = True
                    break  # Restart the loop
            except:
                continue
    
    return U, V, W

def _structured_winograd_attempt(deadline):
    """
    Attempt to construct a lower-rank decomposition using Winograd-style
    cyclic/grouped operations for the 4x4x8 problem.
    
    Strategy: View 4x4x8 as having a block-circulant structure in the p=8 dimension.
    Use Fourier-based grouping to reduce the effective rank.
    """
    # This is an experimental approach - try to find a rank < 98 decomposition
    # by exploiting symmetries in the p=8 dimension
    
    # For now, return None to indicate no improvement found
    # A full Winograd-style construction would require careful algebraic design
    return None, None, None

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
    
    # Try structured search for lower rank
    if time.time() < deadline:
        sU, sV, sW = _structured_winograd_attempt(deadline)
        if sU is not None and sU.shape[1] < best_rank:
            if _verify_decomposition(sU, sV, sW, n, m, p):
                best_U, best_V, best_W = sU, sV, sW
                best_rank = sU.shape[1]
    
    # Try greedy column elimination to reduce rank
    if time.time() < deadline:
        U2, V2, W2 = _greedy_column_elimination(best_U, best_V, best_W, n, m, p, deadline)
        if U2.shape[1] < best_rank and _verify_decomposition(U2, V2, W2, n, m, p):
            best_U, best_V, best_W = U2, V2, W2
            best_rank = U2.shape[1]
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": V.tolist(), "W": W.tolist()}
