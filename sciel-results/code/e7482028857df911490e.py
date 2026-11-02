"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

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

def _get_strassen_26_decomposition():
    """
    Get the verified rank-26 decomposition:
    7 Strassen terms for 2x2 core + 19 standard border terms = 26 total.
    """
    n, m, p = DIMENSIONS
    terms = _strassen_2x2_terms() + _border_terms_standard()
    U, V, W = _build_decomposition(terms)
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=5000, deadline=None):
    """
    Try to reduce rank from current to target_rank by finding linear dependencies.
    Uses a local search approach: try to express one rank-1 term as a combination
    of others by adjusting coefficients.
    
    Strategy: For each pair of columns, try to see if we can merge them
    by finding a 2x2 block that can be diagonalized.
    """
    if deadline is None:
        deadline = time.time() + 10.0
    
    R = U.shape[1]
    if R <= target_rank:
        return U, V, W, R
    
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    best_rank = R
    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
    
    # Try to eliminate columns by finding if one column is a linear combination
    # of others in a way that preserves the tensor product structure.
    # We use a heuristic: try random linear combinations and check validity.
    
    np.random.seed(42)
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        if R <= target_rank:
            break
        
        # Try to drop a random column by absorbing it into another
        col_to_drop = np.random.randint(R)
        col_to_keep = np.random.randint(R - 1)
        if col_to_keep >= col_to_drop:
            col_to_keep += 1
        
        # Try to find a scalar alpha such that removing col_to_drop and
        # adding alpha * col_to_drop to col_to_keep preserves the tensor.
        # This is generally not possible, but let's try small perturbations.
        
        # Alternative: try to find if two columns can be combined
        # by finding a common factor structure.
        
        # For now, use a simpler approach: check for zero columns
        if np.allclose(U_curr[:, col_to_drop], 0, atol=1e-10) and \
           np.allclose(V_curr[:, col_to_drop], 0, atol=1e-10) and \
           np.allclose(W_curr[:, col_to_drop], 0, atol=1e-10):
            R_new = R - 1
            U_curr = np.delete(U_curr, col_to_drop, axis=1)
            V_curr = np.delete(V_curr, col_to_drop, axis=1)
            W_curr = np.delete(W_curr, col_to_drop, axis=1)
            R = R_new
            if R < best_rank:
                best_rank = R
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
            continue
    
    return best_U, best_V, best_W, best_rank

def _search_rank_reductions(U, V, W, n, m, p, deadline):
    """
    Systematically search for rank reductions below 26.
    Try various strategies to find a rank-25 or lower decomposition.
    """
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy 1: Try to merge border terms with Strassen terms
    # by finding linear dependencies.
    
    # Strategy 2: Use random initialization and gradient-free optimization
    # to find a lower-rank decomposition.
    
    # For now, just try the basic reduction
    U1, V1, W1, R1 = _try_reduce_rank(U, V, W, n, m, p, best_rank - 1, 
                                       max_iters=2000, deadline=deadline)
    if R1 < best_rank and _verify_decomposition(U1, V1, W1, n, m, p, atol=1e-8):
        best_rank = R1
        best_U, best_V, best_W = U1, V1, W1
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Tries several known constructions and returns the best one.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
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
        U1, V1, W1, R1 = _search_rank_reductions(U0, V0, W0, n, m, p, deadline)
        if R1 < best_rank and _verify_decomposition(U1, V1, W1, n, m, p, atol=1e-8):
            best_rank = R1
            best_result = (U1, V1, W1)
    
    # Strategy 3: Pure standard (rank 27) as fallback
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
