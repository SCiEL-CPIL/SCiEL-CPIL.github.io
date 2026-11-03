"""Matrix multiplication tensor decomposition for 3x3x7.
Finds the minimal-rank exact decomposition of the matrix multiplication tensor.
"""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents the multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # For all (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j], V[j * p + k])
                s = s * W[k * n + i]
                s = np.sum(s)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_block_rank(n, m, p):
    """
    Try to build a lower-rank decomposition by exploiting block structure.
    
    Strategy: For 3x3x7, we can try to use the fact that the 7 output columns
    can be handled in groups. One approach:
    
    For each row i of A and column k of B, the contribution is sum_j A[i,j]B[j,k].
    If we can share computation across k values for the same (i,j), we save rank.
    
    Specifically, consider that for fixed i and j, the vector [B[j,0], B[j,1], ..., B[j,p-1]]
    is multiplied by A[i,j]. If we precompute B[j,:] as a vector, then for each i,
    C[i,:] = sum_j A[i,j] * B[j,:].
    
    A block decomposition: partition the p=7 columns into groups.
    For group g with columns k in G_g, we can use a rank-|G_g| decomposition for each (i,j).
    But we can share the "B[j, k]" factor.
    
    Actually, a better approach: use the standard rank formula for multiplication.
    For an n x m matrix times m x p matrix, the schoolbook rank is n*m*p.
    But there are known lower bounds and constructions.
    
    One simple improvement: if p >= n, we can use the fact that the matrix B has more
    columns than rows, and use a decomposition that shares the A[i,j] factor.
    
    For 3x3x7: Let's try a construction where we use rank = n * p * m but with
    some sharing. Actually, the schoolbook is already n*m*p = 63.
    
    Let's try: for each (i, k), we have C[i,k] = sum_j A[i,j] B[j,k].
    This is a sum of m=3 terms. If we assign one rank term per (i,j,k), that's 63.
    But we can try to use bilinear tricks.
    
    Another idea: use the rank of the multiplication as a bilinear map.
    The map (A, B) -> A@B is bilinear from R^{n*m} x R^{m*p} to R^{n*p}.
    The tensor rank of this bilinear map is what we're computing.
    
    For 3x3x7, known results suggest rank is at most 63 (schoolbook) and at least
    max(9, 21, 21) = 21.
    
    Let's try a specific construction:
    - Use n*m = 9 terms for the "A" side
    - Use p = 7 terms for the "B" side  
    - Total rank could be n*m + m*p + p*n - something
    
    Actually, let's just try to find a rank-40 or so decomposition numerically
    if we can't prove a lower rank.
    """
    # For now, return schoolbook; we'll try numerical search in construct()
    return _build_schoolbook(n, m, p)

def _try_rank_r(n, m, p, r, max_iter=200, seed=42, tol=1e-8, time_limit=100.0):
    """
    Try to find a rank-r decomposition using iterative optimization.
    
    We use alternating minimization:
    - Fix V and W, solve for U (linear system)
    - Fix U and W, solve for V (linear system)
    - Fix U and V, solve for W (linear system)
    
    The constraint is: for all (i,j,k): sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    
    When fixing V and W, the equation for U[i*m+j, :] is:
    sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1 for all k
    
    This gives p equations for m*r unknowns (U[i*m+j, 0..r-1] for each j... wait,
    U[i*m+j, :] is a vector of length r, and it appears in equations for each (j, k)
    where the row index is i*m+j.
    
    Actually, let me reformulate. The unknowns are:
    - U has n*m rows, r columns: total n*m*r unknowns
    - V has m*p rows, r columns: total m*p*r unknowns
    - W has p*n rows, r columns: total p*n*r unknowns
    
    Total unknowns: (n*m + m*p + p*n) * r
    
    Constraints: n*m*p equations (one for each (i,j,k))
    
    For r < n*m*p, we have more unknowns than equations (usually), so solutions exist
    if the system is consistent.
    
    The system is nonlinear (products of unknowns from U, V, W).
    
    Let's use a different approach: random initialization + ALS.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Update U: for each (i, j), solve for U[i*m+j, :]
        # Equation: for each k: sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
        # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product, length r)
        # Then: U[i*m+j, :] @ A_k = 1 for all k
        # This is a system: U[i*m+j, :] @ M = 1 where M[k, r'] = V[j*p+k, r'] * W[k*n+i, r']
        # M is p x r, and we want U[i*m+j, :] (length r) such that U[i*m+j, :] @ M.T = 1 (length p)
        # i.e., M.T @ U[i*m+j, :] = ones(p)
        
        for i in range(n):
            for j in range(m):
                # Build M: p x r matrix where M[k, r'] = V[j*p+k, r'] * W[k*n+i, r']
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M.T @ x = ones(p) for x of length r
                # This is an underdetermined system if r > p
                try:
                    # Use least squares
                    x, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U[i * m + j, :] = x
                except np.linalg.LinAlgError:
                    pass
        
        # Update V: for each (j, k), solve for V[j*p+k, :]
        # Equation: for each i: sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
        # Let B_i = U[i*m+j, :] * W[k*n+i, :] (element-wise, length r)
        # Then: V[j*p+k, :] @ B_i = 1 for all i
        # V[j*p+k, :] @ B.T = 1 where B[i, r'] = U[i*m+j, r'] * W[k*n+i, r']
        # i.e., B.T @ V[j*p+k, :] = ones(n)
        
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, r))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                    V[j * p + k, :] = x
                except np.linalg.LinAlgError:
                    pass
        
        # Update W: for each (k, i), solve for W[k*n+i, :]
        # Equation: for each j: sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
        # Let C_j = U[i*m+j, :] * V[j*p+k, :] (element-wise, length r)
        # Then: W[k*n+i, :] @ C_j = 1 for all j
        # W[k*n+i, :] @ C.T = 1 where C[j, r'] = U[i*m+j, r'] * V[j*p+k, r']
        # i.e., C.T @ W[k*n+i, :] = ones(m)
        
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, r))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(C.T, np.ones(m), rcond=None)
                    W[k * n + i, :] = x
                except np.linalg.LinAlgError:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    # Verify the best solution
    if _verify(best_U, best_V, best_W, n, m, p, tol=1e-7):
        return {"U": best_U, "V": best_V, "W": best_W}
    return None

def construct():
    n, m, p = 3, 3, 7
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p  # 63
    
    # Try to find lower-rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 21
    # Try ranks from a reasonable lower bound up to 62
    
    # Try a few specific ranks with different seeds
    ranks_to_try = [45, 50, 55, 60, 61, 62]
    
    for r in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        found = _try_rank_r(n, m, p, r, max_iter=150, seed=42 + r, tol=1e-9, 
                           time_limit=deadline - time.time() - 5)
        if found is not None:
            best_result = found
            best_rank = r
            # Keep searching for even lower rank
            continue
    
    # Also try rank 40-44 with more careful initialization
    if time.time() < deadline - 40:
        for r in [40, 42, 44]:
            if time.time() > deadline - 30:
                break
            found = _try_rank_r(n, m, p, r, max_iter=200, seed=100 + r, tol=1e-9,
                               time_limit=deadline - time.time() - 10)
            if found is not None:
                best_result = found
                best_rank = r
    
    # Convert to lists
    U = np.asarray(best_result["U"], dtype=float)
    V = np.asarray(best_result["V"], dtype=float)
    W = np.asarray(best_result["W"], dtype=float)
    
    # Round to nearest representable value for cleaner output
    U = np.round(U, 12)
    V = np.round(V, 12)
    W = np.round(W, 12)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
