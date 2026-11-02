"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_decomposition(rank_terms):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where each dict maps (i,j) or (j,k) or (k,i) to a scalar.
    """
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=atol)

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for the 2x2 core (indices 0,1).
    Each term: (U_dict, V_dict, W_dict)
    """
    terms = []
    
    # m1 = (a00 + a11)(b00 + b11) -> c00 += m1, c11 += m1
    terms.append(({(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1}))
    
    # m2 = (a10 + a11)b00 -> c10 += m2, c11 -= m2
    terms.append(({(1,0): 1, (1,1): 1},
                  {(0,0): 1},
                  {(0,1): 1, (1,1): -1}))
    
    # m3 = a00(b01 - b11) -> c01 += m3, c11 += m3
    terms.append(({(0,0): 1},
                  {(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1}))
    
    # m4 = a11(b10 - b00) -> c00 += m4, c10 += m4
    terms.append(({(1,1): 1},
                  {(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1}))
    
    # m5 = (a00 + a01)b11 -> c00 -= m5, c01 += m5
    terms.append(({(0,0): 1, (0,1): 1},
                  {(1,1): 1},
                  {(0,0): -1, (1,0): 1}))
    
    # m6 = (a10 - a00)(b00 + b01) -> c11 += m6
    terms.append(({(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1},
                  {(1,1): 1}))
    
    # m7 = (a01 - a11)(b10 + b11) -> c00 += m7
    terms.append(({(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1},
                  {(0,0): 1}))
    
    return terms

def _border_terms_standard():
    """
    Return standard multiplication terms for the border (at least one index is 2).
    There are 27 - 8 = 19 such terms.
    """
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _get_strassen_26_decomposition():
    """
    Get the verified rank-26 decomposition:
    7 Strassen terms for 2x2 core + 19 standard border terms = 26 total.
    """
    n, m, p = DIMENSIONS
    terms = _strassen_2x2_terms() + _border_terms_standard()
    U, V, W = _build_decomposition(terms)
    return U, V, W

def _try_standard_decomposition():
    """Standard schoolbook: rank 27."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    U, V, W = _build_decomposition(terms)
    return U, V, W

def _try_25_decomposition():
    """
    Try to construct a rank-25 decomposition by modifying the rank-26 one.
    The idea: if we can find a linear dependency among the 26 terms, we can reduce to 25.
    
    We attempt to find a non-trivial linear combination of the 26 rank-1 tensors that sums to zero.
    This is equivalent to finding a vector (alpha_1, ..., alpha_26) such that
    sum_r alpha_r * (u_r ⊗ v_r ⊗ w_r) = 0.
    
    This is a system of 27 equations (one for each (i,j,k) entry) in 26 unknowns.
    If the system has a non-trivial solution, we can reduce the rank.
    """
    n, m, p = DIMENSIONS
    U, V, W = _get_strassen_26_decomposition()
    R = U.shape[1]
    
    # Build the tensor T_recon = sum_r U[:,r] * V[:,r] * W[:,r]
    # We want to find alpha such that sum_r alpha_r * T_r = 0
    # where T_r is the rank-1 tensor formed by column r of U, V, W
    
    # Reshape
    U_r = U.reshape(n, m, R)
    V_r = V.reshape(m, p, R)
    W_r = W.reshape(p, n, R)
    
    # For each (i,j,k), the equation is:
    # sum_r alpha_r * U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r] = 0
    
    # Build the 27x26 matrix A where A[(i,j,k), r] = U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    A = np.zeros((n*m*p, R))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(R):
                    A[idx, r] = U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                idx += 1
    
    # Check if the null space of A is non-trivial
    # If rank(A) < R, then there exists a non-trivial alpha in the null space
    try:
        rank_A = np.linalg.matrix_rank(A, tol=1e-10)
    except:
        rank_A = R
    
    if rank_A < R:
        # Find a null vector
        # Use SVD to find the right singular vector corresponding to the smallest singular value
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=True)
        # The null vector is the last column of Vt.T
        alpha = Vt[-1, :]
        
        # Check if this is a valid reduction
        # We need to verify that the modified decomposition still works
        # Actually, if A @ alpha = 0, it means the tensor sum is zero, which doesn't help directly.
        # We need a different approach: find two columns that can be merged.
        
        # Alternative: try to find a pair of columns that are "similar" in structure
        # and can be combined into one.
        
        # For now, return None to indicate no simple reduction found
        return None
    
    return None

def _try_gaussian_elimination_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    Uses a more systematic approach: try all pairs and see if they can be merged.
    """
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find if any column is a linear combination of others in a way that
    # preserves the tensor structure. This is hard, so we use a heuristic.
    
    # Heuristic: try to find a pair of columns that have similar support patterns
    # and can be combined.
    
    np.random.seed(42)
    
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        # Try to drop a random column by absorbing it into another
        col_to_drop = np.random.randint(R)
        col_to_keep = np.random.randint(R - 1)
        if col_to_keep >= col_to_drop:
            col_to_keep += 1
        
        # Try to find a scalar alpha such that removing col_to_drop and
        # adding alpha * col_to_drop to col_to_keep preserves the tensor.
        # This requires solving a system of equations.
        
        # For simplicity, just check if the column is already zero
        if np.allclose(U[:, col_to_drop], 0, atol=1e-10) and \
           np.allclose(V[:, col_to_drop], 0, atol=1e-10) and \
           np.allclose(W[:, col_to_drop], 0, atol=1e-10):
            R_new = R - 1
            U_new = np.delete(U, col_to_drop, axis=1)
            V_new = np.delete(V, col_to_drop, axis=1)
            W_new = np.delete(W, col_to_drop, axis=1)
            R = R_new
            if R < best_rank:
                best_rank = R
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Tries several known constructions and returns the best one.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Strassen core + standard border (rank 26)
    if time.time() < deadline:
        U, V, W = _get_strassen_26_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 2: Try to reduce rank via local search
    if best_result is not None and time.time() < deadline:
        U0, V0, W0 = best_result
        U1, V1, W1, R1 = _try_gaussian_elimination_reduction(U0, V0, W0, n, m, p, deadline)
        if R1 < best_rank and _verify_decomposition(U1, V1, W1, n, m, p, atol=1e-8):
            best_rank = R1
            best_result = (U1, V1, W1)
    
    # Strategy 3: Try the rank-25 construction
    if time.time() < deadline:
        result_25 = _try_25_decomposition()
        if result_25 is not None:
            U, V, W = result_25
            if _verify_decomposition(U, V, W, n, m, p):
                R = U.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = (U, V, W)
    
    # Strategy 4: Pure standard (rank 27) as fallback
    if time.time() < deadline:
        U, V, W = _try_standard_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    if best_result is None:
        # Should not happen, but fallback
        U, V, W = _try_standard_decomposition()
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
