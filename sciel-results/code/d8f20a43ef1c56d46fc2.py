"""Matrix multiplication 3x5x5 tensor decomposition via local rank-reduction search."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

def _build_schoolbook_np(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    # Faster: compute tensor T = sum_r U[:,r] V[:,r] W[:,r] and compare
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                    return False
    return True

def _try_reduce_columns(U, V, W, n, m, p, deadline, max_rounds=50):
    """
    Local search: try to remove columns by finding linear dependencies.
    Strategy: for each column r, check if there exists a set S of other columns
    and coefficients such that the rank-1 term r can be absorbed.
    Simplified: try removing column r and see if remaining columns can be
    scaled to still satisfy the tensor (this is too restrictive).
    
    Better: use the fact that if two columns have identical support patterns
    in a certain way, we can combine them.
    
    Most effective simple heuristic: repeatedly try to find a column r such
    that for every (i,j,k) where U[i,j,r]*V[j,k,r]*W[k,i,r] != 0, there exists
    another column s with the same nonzero pattern contribution, allowing cancellation.
    
    Practical approach: try random linear combinations of 2-3 columns to replace
    one column, checking if the tensor is preserved.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    n, m, p = n, m, p
    rank = U.shape[1]
    
    def tensor_check(U, V, W):
        """Return the full tensor as a flat array for comparison."""
        r = U.shape[1]
        T = np.zeros((n*m, m*p, p*n))
        for c in range(r):
            T += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return T
    
    target_T = tensor_check(U, V, W)
    
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and time.time() < deadline - 15:
        improved = False
        rounds += 1
        rank = U.shape[1]
        # Try each column for removal via absorption
        for r in range(rank):
            if time.time() > deadline - 15:
                return U, V, W
            # Check if column r is "redundant" - try to express its contribution
            # as a combination of others
            # Simple test: zero out column r, compute residual, see if it's
            # already covered by other columns (i.e., residual is zero)
            U_save = U[:, r].copy()
            V_save = V[:, r].copy()
            W_save = W[:, r].copy()
            U[:, r] = 0
            V[:, r] = 0
            W[:, r] = 0
            
            T_now = tensor_check(U, V, W)
            if np.allclose(T_now, target_T, atol=1e-9):
                # Column r was redundant
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                improved = True
                break
            else:
                U[:, r] = U_save
                V[:, r] = V_save
                W[:, r] = W_save
    
    return U, V, W

def _try_strassen_blocks(n, m, p, deadline):
    """
    Apply Strassen to 2x2x2 sub-blocks where possible to reduce rank.
    For 3x5x5: we can identify 2x2x2 blocks and apply Strassen.
    
    Split into blocks of size 2 in each dimension where possible.
    A is 3x5, B is 5x5, C is 3x5.
    
    We can form 2x2x2 blocks:
    - i in {0,1}, j in {0,1}, k in {0,1}
    - i in {0,1}, j in {0,1}, k in {2,3}
    - i in {0,1}, j in {0,1}, k in {4}  -- not 2x2
    - i in {0,1}, j in {2,3}, k in {0,1}
    - etc.
    
    Let's just try all possible 2x2x2 sub-blocks and apply Strassen to each.
    """
    # Start with schoolbook
    U, V, W = _build_schoolbook_np(n, m, p)
    
    # Identify all 2x2x2 blocks
    # A block is defined by (i0, j0, k0) where i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    # such that i0+1 < n, j0+1 < m, k0+1 < p
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                blocks.append((i0, j0, k0))
    
    # For each block, the schoolbook terms are at specific column indices
    # In schoolbook, column index for (i,j,k) is i*m + j)*p + k ... let me recompute
    # Actually in _build_schoolbook_np, col = i*m*p + j*p + k
    
    # For a 2x2x2 block at (i0, j0, k0), the 8 schoolbook terms are:
    # (i0+j0*p+k0), (i0+j0+1)*p+k0, ... etc
    # Let's map them:
    # term(i,j,k) column = i*m*p + j*p + k
    
    # Apply Strassen to each block sequentially
    for (i0, j0, k0) in blocks:
        if time.time() > deadline - 30:
            break
        # The 8 terms for this block
        indices = []
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    col = i * m * p + j * p + k
                    indices.append(col)
        
        # Check if all 8 columns exist and are still "active" (non-zero in U,V,W)
        # For now, just apply Strassen replacement
        # The 8 schoolbook terms become 7 Strassen terms
        # We need to modify U, V, W in place
        
        # Strassen for 2x2x2:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 -= M1, C10 -= M1, C11 += M1
        # M2 = (A10+A11)B00 -> C10 += M2, C11 += M2
        # M3 = A00(B10-B11) -> C00 += M3, C10 += M3
        # M4 = A11(B01-B00) -> C01 += M4, C11 += M4
        # M5 = (A00+A01)B11 -> C01 += M5, C11 += M5
        # M6 = (A10-A00)(B00+B01) -> C00 += M6, C01 += M6
        # M7 = (A01-A10)(B10+B11) -> C00 += M7, C10 += M7
        
        # In terms of the 8 original columns:
        # col000 = A00*B00*C00 (i0,j0,k0)
        # col001 = A00*B00*C01 (i0,j0,k0+1)
        # col010 = A00*B01*C00 (i0,j0+1,k0)
        # col011 = A00*B01*C01 (i0,j0+1,k0+1)
        # col100 = A10*B00*C00 (i0+1,j0,k0)
        # col101 = A10*B00*C01 (i0+1,j0,k0+1)
        # col110 = A10*B01*C00 (i0+1,j0+1,k0)
        # col111 = A10*B01*C01 (i0+1,j0+1,k0+1)
        
        c000, c001, c010, c011, c100, c101, c110, c111 = indices
        
        # For each Strassen term, we need to add contributions to U, V, W
        # and subtract the original 8 terms
        
        # This is complex to do in-place without breaking other blocks.
        # Skip this for now and rely on column removal.
        pass
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook_np(n, m, p)
    best_rank = U.shape[1]
    
    # Try column removal
    if time.time() < deadline - 30:
        U, V, W = _try_reduce_columns(U, V, W, n, m, p, deadline)
        if _verify(U, V, W, n, m, p):
            best_rank = U.shape[1]
    
    # Try Strassen blocks
    if time.time() < deadline - 30:
        U2, V2, W2 = _try_strassen_blocks(n, m, p, deadline)
        if _verify(U2, V2, W2, n, m, p):
            if U2.shape[1] < best_rank:
                U, V, W = U2, V2, W2
                best_rank = U.shape[1]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
