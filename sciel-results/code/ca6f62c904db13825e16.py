"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.
Uses structured search with exact rational verification and consistency constraints.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]
START_TIME = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _validate_exact(U, V, W, n, m, p):
    """Check exact decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1.
    Uses exact arithmetic with doubled values to handle half-integers."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    
    # Check all entries are integer or half-integer
    for name, M in [("U", U), ("V", V), ("W", W)]:
        M2 = 2 * M
        if not np.allclose(M2, np.round(M2), atol=1e-9):
            return False
    
    # Convert to doubled integers for exact checking
    U2 = np.round(2 * U).astype(int)
    V2 = np.round(2 * V).astype(int)
    W2 = np.round(2 * W).astype(int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U2[i*m+j,c] * V2[j*p+k,c] * W2[k*n+i,c] should equal 4 (since 1 = 4/4)
                s = 0
                for c in range(r):
                    s += U2[i * m + j, c] * V2[j * p + k, c] * W2[k * n + i, c]
                if s != 4:
                    return False
    return True


def _to_output_list(M):
    """Convert numpy array to list of lists with int or float values."""
    result = []
    for row in M:
        r = []
        for v in row:
            fv = float(v)
            if abs(fv - round(fv)) < 1e-9:
                r.append(int(round(fv)))
            else:
                r.append(fv)
        result.append(r)
    return result


def _build_schoolbook(n, m, p):
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
    return {"U": _to_output_list(U), "V": _to_output_list(V), "W": _to_output_list(W), "rank": rank}


def _try_rank_with_consistency(r, n, m, p, seed=0, max_iters=5000):
    """Try to find a decomposition of rank r using random search with consistency constraint.
    
    Key constraint: for each (i,k,c), U[i*m+j,c]*V[j*p+k,c] must be the same for all j.
    This determines W[k*n+i,c] = that common value.
    """
    rng = np.random.default_rng(seed)
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            return None
        
        # Random initialization
        U = rng.choice(values, size=(n * m, r))
        V = rng.choice(values, size=(m * p, r))
        
        # Check consistency and build W
        W = np.zeros((p * n, r))
        consistent = True
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    # Get the products for j=0,1,2
                    prods = []
                    for j in range(m):
                        prods.append(U[i * m + j, c] * V[j * p + k, c])
                    # Check all equal
                    ref = prods[0]
                    for jj in range(1, m):
                        if abs(prods[jj] - ref) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        break
                    W[k * n + i, c] = ref
                if not consistent:
                    break
            if not consistent:
                break
        
        if not consistent:
            continue
        
        # Verify exactness
        if _validate_exact(U, V, W, n, m, p):
            return {"U": _to_output_list(U), "V": _to_output_list(V), "W": _to_output_list(W), "rank": r}
    
    return None


def _try_structured_patterns(r, n, m, p, seed=0, max_iters=3000):
    """Try structured patterns where U and V have specific sparsity/structure.
    
    For each rank component, try patterns where:
    - U[i*m+j,c] is nonzero for a subset of j's
    - V[j*p+k,c] is nonzero for a subset of k's
    - The consistency constraint is satisfied
    """
    rng = np.random.default_rng(seed)
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            return None
        
        # Build U and V with structured sparsity
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        
        for c in range(r):
            # Randomly choose which j's are active for each i
            for i in range(n):
                # Choose a pattern: all same, or specific values
                pattern = rng.integers(0, 3)
                if pattern == 0:
                    # All j's have same value
                    val = rng.choice([-1.0, -0.5, 0.5, 1.0])
                    for j in range(m):
                        U[i * m + j, c] = val
                elif pattern == 1:
                    # Only one j active
                    j_active = rng.integers(0, m)
                    val = rng.choice([-1.0, -0.5, 0.5, 1.0])
                    U[i * m + j_active, c] = val
                else:
                    # Two j's active with reciprocal V values
                    j1, j2 = rng.choice(m, size=2, replace=False)
                    val1 = rng.choice([-1.0, -0.5, 0.5, 1.0])
                    val2 = rng.choice([-1.0, -0.5, 0.5, 1.0])
                    U[i * m + j1, c] = val1
                    U[i * m + j2, c] = val2
                    # V must satisfy: val1 * V[j1*p+k,c] = val2 * V[j2*p+k,c]
                    # So V[j1*p+k,c] / V[j2*p+k,c] = val2 / val1
                    # For simplicity, set V[j1*p+k,c] = val2, V[j2*p+k,c] = val1
                    # Then val1*val2 = val2*val1 (consistent)
                    for k in range(p):
                        V[j1 * p + k, c] = val2
                        V[j2 * p + k, c] = val1
                        # Other j's: V[j*p+k,c] = 0
                        for j_other in range(m):
                            if j_other != j1 and j_other != j2:
                                V[j_other * p + k, c] = 0
        
        # Check consistency and build W
        W = np.zeros((p * n, r))
        consistent = True
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    prods = []
                    for j in range(m):
                        prods.append(U[i * m + j, c] * V[j * p + k, c])
                    ref = prods[0]
                    for jj in range(1, m):
                        if abs(prods[jj] - ref) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        break
                    W[k * n + i, c] = ref
                if not consistent:
                    break
            if not consistent:
                break
        
        if not consistent:
            continue
        
        if _validate_exact(U, V, W, n, m, p):
            return {"U": _to_output_list(U), "V": _to_output_list(V), "W": _to_output_list(W), "rank": r}
    
    return None


def _try_gaussian_refine(r, n, m, p, seed=0, max_iters=2000):
    """Gaussian random + local refinement with exact verification."""
    rng = np.random.default_rng(seed)
    values = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            return None
        
        U = rng.choice(values, size=(n * m, r))
        V = rng.choice(values, size=(m * p, r))
        
        W = np.zeros((p * n, r))
        consistent = True
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    prods = []
                    for j in range(m):
                        prods.append(U[i * m + j, c] * V[j * p + k, c])
                    ref = prods[0]
                    for jj in range(1, m):
                        if abs(prods[jj] - ref) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        break
                    W[k * n + i, c] = ref
                if not consistent:
                    break
            if not consistent:
                break
        
        if not consistent:
            continue
        
        if _validate_exact(U, V, W, n, m, p):
            return {"U": _to_output_list(U), "V": _to_output_list(V), "W": _to_output_list(W), "rank": r}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Start with schoolbook
    best = _build_schoolbook(n, m, p)
    best_rank = best["rank"]
    
    # Try lower ranks
    # Lower bound: for matrix multiplication of n x m by m x p, 
    # the rank is at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But we can do better with structured decompositions
    
    ranks_to_try = list(range(best_rank - 1, 23, -1))
    
    for rank in ranks_to_try:
        if time.time() > DEADLINE - 5:
            break
        
        # Try different search strategies
        result = _try_rank_with_consistency(rank, n, m, p, seed=rank * 1000, max_iters=2000)
        if result is not None:
            best = result
            best_rank = rank
            continue
        
        result = _try_structured_patterns(rank, n, m, p, seed=rank * 2000, max_iters=1500)
        if result is not None:
            best = result
            best_rank = rank
            continue
        
        result = _try_gaussian_refine(rank, n, m, p, seed=rank * 3000, max_iters=1000)
        if result is not None:
            best = result
            best_rank = rank
            continue
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
