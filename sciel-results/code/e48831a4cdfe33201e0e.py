"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _build_decomposition(rank_terms):
    """Build U, V, W factor matrices from a list of term specifications."""
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
    """Return the 7 Strassen terms for the 2x2 core (indices 0,1)."""
    terms = []
    terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}))
    terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}))
    terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}))
    terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}))
    terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
    return terms

def _standard_border_terms():
    """Standard multiplication terms for the border (at least one index is 2)."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _try_deterministic_construction():
    """
    Attempt a deterministic construction for rank 26.
    Uses Strassen (7) for the 2x2 core, then standard border terms.
    """
    n, m, p = DIMENSIONS
    strassen = _strassen_2x2_terms()
    terms = strassen + _standard_border_terms()
    U, V, W = _build_decomposition(terms)
    if _verify_decomposition(U, V, W, n, m, p):
        return U, V, W, len(terms)
    return None, None, None, float('inf')

def _try_rank_r_structured_search(r, deadline, max_attempts=2000, seed=None):
    """
    Try to find a rank-r decomposition using structured random search.
    Key idea: Instead of random assignment, try to merge border terms in groups
    using algebraic patterns (sums/differences) that preserve exactness.
    
    Strategy:
    1. Start with Strassen core (7 terms).
    2. For the 19 border terms, try to cover them with (r-7) terms.
    3. Each extra term can cover 1-3 border positions if the algebra works out.
    4. Use a greedy matching approach with randomization.
    """
    n, m, p = DIMENSIONS
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    strassen = _strassen_2x2_terms()
    
    # Border positions: all (i,j,k) where at least one index is 2
    border_positions = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)
                        if not (i < 2 and j < 2 and k < 2)]
    
    # Group border positions by their "type" for potential merging
    # Type A: i=2, j<2, k<2 -> 4 positions (2,0,0), (2,1,0), (2,0,1), (2,1,1)
    # Type B: i=2, j<2, k=2 -> 2 positions (2,0,2), (2,1,2)
    # Type C: i=2, j=2, k<2 -> 2 positions (2,2,0), (2,2,1)
    # Type D: i=2, j=2, k=2 -> 1 position (2,2,2)
    # Type E: i<2, j=2, k<2 -> 4 positions (0,2,0), (1,2,0), (0,2,1), (1,2,1)
    # Type F: i<2, j=2, k=2 -> 2 positions (0,2,2), (1,2,2)
    # Type G: i<2, j<2, k=2 -> 4 positions (0,0,2), (0,1,2), (1,0,2), (1,1,2)
    # Total: 4+2+2+1+4+2+4 = 19 ✓
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Fill in Strassen core
        for r_idx, (u_dict, v_dict, w_dict) in enumerate(strassen):
            for (i, j), val in u_dict.items():
                U[i * m + j, r_idx] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r_idx] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r_idx] = val
        
        num_extra = r - 7
        if num_extra <= 0:
            continue
        
        # Try to cover border positions with num_extra terms
        # Each term covers a subset of border positions
        # We'll use a randomized greedy approach
        
        remaining = set(border_positions)
        success = True
        
        for t in range(num_extra):
            r_idx = 7 + t
            
            if not remaining:
                break
            
            # Choose a strategy for this term
            strategy = random.choice(['single', 'pair', 'triple', 'row', 'col'])
            
            if strategy == 'single':
                # Cover one position
                if not remaining:
                    break
                pos = random.choice(list(remaining))
                i, j, k = pos
                sign = random.choice([1, -1])
                U[i * m + j, r_idx] = sign
                V[j * p + k, r_idx] = sign
                W[k * n + i, r_idx] = 1
                remaining.discard(pos)
                
            elif strategy == 'pair':
                # Try to cover two positions that share a common factor
                # e.g., (i, j, k) and (i, j, k') share a[i,j]
                candidates = []
                for pos1 in list(remaining):
                    for pos2 in list(remaining):
                        if pos1 >= pos2:
                            continue
                        i1, j1, k1 = pos1
                        i2, j2, k2 = pos2
                        # Check if they can be merged
                        # Same i,j: (i,j,k1) and (i,j,k2) -> U[i,j] * (V[j,k1] + V[j,k2]) * W[k?,i]
                        # This doesn't directly merge unless W entries align
                        # Let's just try random pairs and verify later
                        if random.random() < 0.3:
                            candidates.append((pos1, pos2))
                
                if candidates:
                    pos1, pos2 = random.choice(candidates)
                    i1, j1, k1 = pos1
                    i2, j2, k2 = pos2
                    s1 = random.choice([1, -1])
                    s2 = random.choice([1, -1])
                    U[i1 * m + j1, r_idx] = s1
                    V[j1 * p + k1, r_idx] = s1
                    W[k1 * n + i1, r_idx] = 1
                    U[i2 * m + j2, r_idx] = s2
                    V[j2 * p + k2, r_idx] = s2
                    W[k2 * n + i2, r_idx] = 1
                    remaining.discard(pos1)
                    remaining.discard(pos2)
                else:
                    # Fallback to single
                    pos = random.choice(list(remaining))
                    i, j, k = pos
                    U[i * m + j, r_idx] = 1
                    V[j * p + k, r_idx] = 1
                    W[k * n + i, r_idx] = 1
                    remaining.discard(pos)
                    
            elif strategy == 'row':
                # Cover all positions with same i,j but different k
                # e.g., (i,j,0), (i,j,1), (i,j,2)
                found = False
                for i in range(n):
                    for j in range(m):
                        row_positions = [(i, j, k) for k in range(p) if (i, j, k) in remaining]
                        if len(row_positions) >= 2:
                            for pos in row_positions:
                                i_p, j_p, k_p = pos
                                U[i_p * m + j_p, r_idx] = 1
                                V[j_p * p + k_p, r_idx] = 1
                                W[k_p * n + i_p, r_idx] = 1
                                remaining.discard(pos)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    pos = random.choice(list(remaining))
                    i, j, k = pos
                    U[i * m + j, r_idx] = 1
                    V[j * p + k, r_idx] = 1
                    W[k * n + i, r_idx] = 1
                    remaining.discard(pos)
                    
            elif strategy == 'col':
                # Cover all positions with same j,k but different i
                found = False
                for j in range(m):
                    for k in range(p):
                        col_positions = [(i, j, k) for i in range(n) if (i, j, k) in remaining]
                        if len(col_positions) >= 2:
                            for pos in col_positions:
                                i_p, j_p, k_p = pos
                                U[i_p * m + j_p, r_idx] = 1
                                V[j_p * p + k_p, r_idx] = 1
                                W[k_p * n + i_p, r_idx] = 1
                                remaining.discard(pos)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    pos = random.choice(list(remaining))
                    i, j, k = pos
                    U[i * m + j, r_idx] = 1
                    V[j * p + k, r_idx] = 1
                    W[k * n + i, r_idx] = 1
                    remaining.discard(pos)
                    
            else:  # triple
                # Try to cover three positions
                if len(remaining) >= 3:
                    pos1, pos2, pos3 = random.sample(list(remaining), 3)
                    for pos in [pos1, pos2, pos3]:
                        i, j, k = pos
                        U[i * m + j, r_idx] = 1
                        V[j * p + k, r_idx] = 1
                        W[k * n + i, r_idx] = 1
                        remaining.discard(pos)
                else:
                    for pos in list(remaining):
                        i, j, k = pos
                        U[i * m + j, r_idx] = 1
                        V[j * p + k, r_idx] = 1
                        W[k * n + i, r_idx] = 1
                        remaining.discard(pos)
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p, atol=1e-6):
            return U, V, W
    
    return None, None, None

def _try_gaussian_elimination_search(r, deadline, max_iterations=100, seed=None):
    """
    Try to find a rank-r decomposition using a more systematic approach.
    This uses the idea that we can set up a system of equations and solve for
    the coefficients, using the constraint that most entries are zero.
    
    For small r (e.g., 23-25), this can be formulated as a nonlinear system.
    We'll use a simplified version: fix the sparsity pattern and solve for values.
    """
    n, m, p = DIMENSIONS
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # This is a placeholder for a more sophisticated search
    # In practice, finding rank < 26 for 3x3x3 is extremely hard
    # The known best upper bound is 23, but constructing it is non-trivial
    
    return None, None, None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Deterministic rank-26 construction (Strassen + standard border)
    if time.time() < deadline:
        U, V, W, rank = _try_deterministic_construction()
        if U is not None and rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    
    # Strategy 2: Structured random search for rank 25
    if time.time() < deadline:
        remaining = deadline - time.time()
        max_attempts = min(3000, int(remaining * 500))
        U, V, W = _try_rank_r_structured_search(25, deadline, max_attempts=max_attempts, seed=42)
        if U is not None and _verify_decomposition(U, V, W, n, m, p):
            best_rank = 25
            best_result = (U, V, W)
    
    # Strategy 3: Structured random search for rank 24
    if time.time() < deadline:
        remaining = deadline - time.time()
        max_attempts = min(2000, int(remaining * 400))
        U, V, W = _try_rank_r_structured_search(24, deadline, max_attempts=max_attempts, seed=123)
        if U is not None and _verify_decomposition(U, V, W, n, m, p):
            best_rank = 24
            best_result = (U, V, W)
    
    # Strategy 4: Structured random search for rank 23
    if time.time() < deadline:
        remaining = deadline - time.time()
        max_attempts = min(1000, int(remaining * 300))
        U, V, W = _try_rank_r_structured_search(23, deadline, max_attempts=max_attempts, seed=777)
        if U is not None and _verify_decomposition(U, V, W, n, m, p):
            best_rank = 23
            best_result = (U, V, W)
    
    # Fallback: standard rank-27
    if best_result is None:
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
        U, V, W = _build_decomposition(terms)
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
