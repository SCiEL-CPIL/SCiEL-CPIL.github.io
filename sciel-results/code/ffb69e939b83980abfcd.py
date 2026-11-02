import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _find_dependencies(U, V, W, rank, tol=1e-9):
    """
    Find exact integer linear dependencies among rank-1 terms.
    Returns list of indices of dependent terms that can be removed.
    """
    # For each pair of terms, check if one is a rational multiple of the other
    # More generally, check if a term can be expressed as a linear combination of others
    # This is expensive, so we use a heuristic approach
    
    # First, check for simple cases: duplicate or negated terms
    dependent = set()
    
    for i in range(rank):
        if i in dependent:
            continue
        for j in range(i+1, rank):
            if j in dependent:
                continue
            # Check if term j is a scalar multiple of term i
            # Compare non-zero patterns
            Ui = U[:, i]
            Uj = U[:, j]
            Vi = V[:, i]
            Vj = V[:, j]
            Wi = W[:, i]
            Wj = W[:, j]
            
            # Find first non-zero in each
            nz_i_u = np.nonzero(Ui)[0]
            nz_j_u = np.nonzero(Uj)[0]
            
            if len(nz_i_u) == 0 or len(nz_j_u) == 0:
                continue
            
            # Check if support sets are the same
            if set(nz_i_u) != set(nz_j_u):
                continue
            
            # Check if ratio is constant
            ratio_u = None
            for idx in nz_i_u:
                if Ui[idx] != 0:
                    r = Uj[idx] / Ui[idx]
                    if ratio_u is None:
                        ratio_u = r
                    elif abs(r - ratio_u) > tol:
                        ratio_u = None
                        break
            
            if ratio_u is None or abs(ratio_u) < tol:
                continue
            
            # Verify with V and W
            nz_i_v = np.nonzero(Vi)[0]
            if len(nz_i_v) == 0:
                continue
            if set(nz_i_v) != set(np.nonzero(Vj)[0]):
                continue
            
            ratio_v = None
            for idx in nz_i_v:
                if Vi[idx] != 0:
                    r = Vj[idx] / Vi[idx]
                    if ratio_v is None:
                        ratio_v = r
                    elif abs(r - ratio_u) > tol:
                        ratio_v = None
                        break
            
            if ratio_v is None or abs(ratio_v - ratio_u) > tol:
                continue
            
            nz_i_w = np.nonzero(Wi)[0]
            if len(nz_i_w) == 0:
                continue
            if set(nz_i_w) != set(np.nonzero(Wj)[0]):
                continue
            
            ratio_w = None
            for idx in nz_i_w:
                if Wi[idx] != 0:
                    r = Wj[idx] / Wi[idx]
                    if ratio_w is None:
                        ratio_w = r
                    elif abs(r - ratio_u) > tol:
                        ratio_w = None
                        break
            
            if ratio_w is not None and abs(ratio_w - ratio_u) < tol:
                dependent.add(j)
    
    return list(dependent)

def _try_reduce_by_recombination(U, V, W, n, m, p, max_iterations=200):
    """
    Try to reduce rank by finding linear combinations that simplify the decomposition.
    Uses a greedy approach: try to express one term as a combination of others.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    # Try removing each term and see if we can compensate with small adjustments
    # This is a heuristic - we'll try random linear combinations
    
    import random
    random.seed(42)
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for attempt in range(max_iterations):
        # Pick a random subset of terms to try to eliminate
        if rank <= 40:
            break
        
        # Try to find if any term is in the span of others
        # Use a simplified approach: check if removing term r and adjusting
        # other terms can still produce the same tensor
        
        # For now, just try simple dependency elimination
        deps = _find_dependencies(U, V, W, rank)
        if deps:
            # Remove dependent terms
            keep = [i for i in range(rank) if i not in deps]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            rank = len(keep)
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist()

def _construct_strassen_decomposition(n, m, p, deadline):
    """
    Construct the rank-50 Strassen-based decomposition.
    """
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        # M2
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1
        # M3
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1
        # M4
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        # M5
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1
        # M6
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        # M7
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return
        
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1
        col += 1
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1
        col += 1
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        add_schoolbook_221(0, j_start, 6)
    
    if col != rank:
        return None
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # First, construct the rank-50 decomposition
    result = _construct_strassen_decomposition(n, m, p, deadline)
    if result is None:
        return _schoolbook(n, m, p)
    
    U, V, W = result
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        return _schoolbook(n, m, p)
    
    # Try to reduce rank by finding dependencies
    if time.time() < deadline - 20:
        U_red, V_red, W_red = _try_reduce_by_recombination(U, V, W, n, m, p, max_iterations=100)
        
        # Verify reduced decomposition
        if _verify(U_red, V_red, W_red, n, m, p):
            return {"U": U_red, "V": V_red, "W": W_red}
    
    # Return the original rank-50 decomposition
    return {"U": U, "V": V, "W": W}
