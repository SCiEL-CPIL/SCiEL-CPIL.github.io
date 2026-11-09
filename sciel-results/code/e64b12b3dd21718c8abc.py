"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0, 1)
                for c in range(rank):
                    s += Fraction(str(U[i*m+j][c])) * Fraction(str(V[j*p+k][c])) * Fraction(str(W[k*n+i][c]))
                if s != Fraction(1, 1):
                    return False
    return True

def _schoolbook(n, m, p):
    """Create the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _structured_search(n, m, p, target_rank, max_time=30.0, seed=42):
    """
    Structured search for low-rank decomposition.
    Uses the fact that matrix multiplication has a specific tensor structure.
    Each rank term corresponds to a "multiplication pattern".
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    best = None
    
    for attempt in range(500):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with small random values
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Strategy: use a "factorized" approach where each rank term
        # corresponds to a rank-1 update pattern
        for c in range(target_rank):
            # Choose random "support" patterns
            # Each term should contribute to a structured subset
            u_support = np.zeros(nm)
            v_support = np.zeros(mp)
            w_support = np.zeros(pn)
            
            # Randomly select which (i,j) pairs this term affects in U
            num_u = random.randint(1, min(4, nm))
            u_indices = random.sample(range(nm), num_u)
            for idx in u_indices:
                u_support[idx] = random.choice([-1, 1])
            
            # For V, select (j,k) pairs that are "compatible" with U's support
            # (i.e., share some j values)
            used_js = set()
            for idx in u_indices:
                used_js.add(idx // p)  # j index from i*m+j
            # Actually j = idx % m for U[i*m+j]
            used_js = set()
            for idx in u_indices:
                used_js.add(idx % m)
            
            num_v = random.randint(1, min(7, mp))
            v_indices = random.sample(range(mp), num_v)
            for idx in v_indices:
                v_support[idx] = random.choice([-1, 1])
            
            # For W, select (k,i) pairs
            num_w = random.randint(1, min(6, pn))
            w_indices = random.sample(range(pn), num_w)
            for idx in w_indices:
                w_support[idx] = random.choice([-1, 1])
            
            U[:, c] = u_support
            V[:, c] = v_support
            W[:, c] = w_support
        
        # Check if this gives a valid decomposition
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Try to clean up to get integer/half-integer values
            U_list = [[float(x) for x in row] for row in U.tolist()]
            V_list = [[float(x) for x in row] for row in V.tolist()]
            W_list = [[float(x) for x in row] for row in W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_linear_reduction(U, V, W, n, m, p, max_time=30.0):
    """
    Try to reduce the rank by finding linear dependencies.
    Given a decomposition of rank r, try to find a smaller rank r' < r
    by expressing some columns as linear combinations of others.
    """
    start_time = time.time()
    rank = len(U[0])
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    best_rank = rank
    best_result = None
    
    # Try to find if any column is linearly dependent on others
    # For each column c, check if it can be expressed as a combination of others
    for target_rank in range(max(28, rank - 10), rank):
        if time.time() - start_time > max_time:
            break
        
        # Try random subsets of columns to keep
        for trial in range(100):
            if time.time() - start_time > max_time:
                break
            
            # Choose which columns to keep
            keep_cols = random.sample(range(rank), target_rank)
            keep_cols.sort()
            
            U_sub = U_np[:, keep_cols]
            V_sub = V_np[:, keep_cols]
            W_sub = W_np[:, keep_cols]
            
            # Check if this sub-decomposition still works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U_sub[i*m+j, c] * V_sub[j*p+k, c] * W_sub[k*n+i, c]
                        if abs(s - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {
                        "U": U_sub.tolist(),
                        "V": V_sub.tolist(),
                        "W": W_sub.tolist()
                    }
    
    return best_result, best_rank

def _build_decomposition_from_factors(n, m, p, factors):
    """
    Build a decomposition from a list of factor specifications.
    Each factor is a tuple (U_pattern, V_pattern, W_pattern) where each pattern
    is a dict mapping (row_index) -> value.
    """
    rank = len(factors)
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    
    for c, (u_pat, v_pat, w_pat) in enumerate(factors):
        for idx, val in u_pat.items():
            U[idx][c] = val
        for idx, val in v_pat.items():
            V[idx][c] = val
        for idx, val in w_pat.items():
            W[idx][c] = val
    
    return U, V, W

def _try_kronecker_based(n, m, p, max_time=30.0):
    """
    Try to build a decomposition using Kronecker product structure.
    For 3x4x7, we can try to exploit the factorization of dimensions.
    """
    start_time = time.time()
    
    # One approach: decompose 4 = 2*2, 7 stays as 7, 3 stays as 3
    # Then use Strassen-like ideas on the 2x2 blocks
    
    # Alternative: use the fact that we can group columns
    # For each i in [0,3), j in [0,4), k in [0,7)
    # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    # Try a "block" approach where we group j values
    # Since m=4, we can try 2x2 blocks
    
    # For now, try a systematic construction
    # Each rank term will handle a specific (i, j_group, k) pattern
    
    factors = []
    
    # Create factors for each i, j, k combination but with sharing
    # Key insight: for fixed i and k, varying j, we can potentially share structure
    
    for i in range(n):
        for k in range(p):
            # For this (i,k) pair, we need to handle all j in [0,4)
            # Use 4 separate terms or try to combine
            for j in range(m):
                u_pat = {i * m + j: 1.0}
                v_pat = {j * p + k: 1.0}
                w_pat = {k * n + i: 1.0}
                factors.append((u_pat, v_pat, w_pat))
    
    U, V, W = _build_decomposition_from_factors(n, m, p, factors)
    
    # Verify
    rank = len(factors)
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(s - 1.0) > 1e-8:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Leave 5s safety margin
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Strategy 1: Try structured search for lower rank
    for target_rank in range(50, 84):
        if time.time() > deadline - 30:
            break
        
        result = _structured_search(n, m, p, target_rank, max_time=15.0, seed=target_rank)
        if result is not None:
            if target_rank < best_rank:
                best_result = result
                best_rank = target_rank
                break  # Found a valid one, try to go lower
    
    if time.time() > deadline - 20:
        return best_result
    
    # Strategy 2: Try linear reduction from schoolbook
    result, reduced_rank = _try_linear_reduction(U, V, W, n, m, p, max_time=20.0)
    if result is not None and reduced_rank < best_rank:
        best_result = result
        best_rank = reduced_rank
    
    if time.time() > deadline - 10:
        return best_result
    
    # Strategy 3: Try Kronecker-based construction
    result = _try_kronecker_based(n, m, p, max_time=15.0)
    if result is not None:
        rank = len(result["U"][0])
        if rank < best_rank:
            best_result = result
            best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
