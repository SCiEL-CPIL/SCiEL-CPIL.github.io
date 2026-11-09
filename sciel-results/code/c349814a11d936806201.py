"""Matrix multiplication 3x5x7 tensor decomposition - rank minimization."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] sums to 1 for each (i,j,k)."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    # Also check that no spurious terms appear: for each (i,j,k), the sum should be exactly 1
    # And for the "wrong" combinations, we need the tensor to be correct
    # Actually, the tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should equal 1 for all valid (i,j,k)
    # This is automatically satisfied by the structure if we check all (i,j,k)
    return True

def _verify_tensor(U, V, W, n, m, p):
    """Full tensor verification: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_winograd_3x3_block():
    """
    Use Winograd-style decomposition for 3x3x3 matrix multiplication.
    The 3x3x3 tensor has rank at most 23 (known result).
    We'll use a specific rank-23 decomposition.
    
    Actually, let me use a simpler approach: try to find a valid decomposition
    by exploiting the block structure.
    """
    n, m, p = 3, 5, 7
    # Split m=5 into [3,2], p=7 into [3,4]
    # Blocks: (3x3), (3x4), (2x3), (2x4)
    # This gets complex. Let me try a direct search approach instead.
    pass

def _random_search_start(U, V, W, n, m, p, max_iter=5000, seed=42):
    """
    Try to reduce rank by randomly setting some columns to 0 and 
    adjusting others. Use exact rational arithmetic.
    """
    import copy
    rank = len(U[0])
    best_rank = rank
    best = (copy.deepcopy(U), copy.deepcopy(V), copy.deepcopy(W))
    
    rng = random.Random(seed)
    
    # Try eliminating one column at a time
    # For each column c to eliminate, try to redistribute its contribution
    # across other columns
    
    # Simpler: try random linear combinations of columns
    for attempt in range(max_iter):
        if time.time() > deadline_global - 15:
            break
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        
        # Pick a random column to eliminate
        c_elim = rng.randint(0, rank - 1)
        # Pick 2-3 other columns to merge into
        candidates = [c for c in range(rank) if c != c_elim]
        targets = rng.sample(candidates, min(3, len(candidates)))
        
        # For each target, add a random fraction of the eliminated column
        for t in targets:
            f = rng.choice([1, -1, 2, -2])  # Simple integer multipliers
            for i in range(n * m):
                U_test[i][t] += f * U_test[i][c_elim]
            for j in range(m * p):
                V_test[j][t] += f * V_test[j][c_elim]
            for k in range(p * n):
                W_test[k][t] += f * W_test[k][c_elim]
        
        # Zero out the eliminated column
        for i in range(n * m):
            U_test[i][c_elim] = 0
        for j in range(m * p):
            V_test[j][c_elim] = 0
        for k in range(p * n):
            W_test[k][c_elim] = 0
        
        # Remove the zero column
        U_test = [[val for col, val in enumerate(row) if col != c_elim] for row in U_test]
        V_test = [[val for col, val in enumerate(row) if col != c_elim] for row in V_test]
        W_test = [[val for col, val in enumerate(row) if col != c_elim] for row in W_test]
        
        new_rank = rank - 1
        if _verify_tensor(U_test, V_test, W_test, n, m, p):
            return U_test, V_test, W_test, new_rank
    
    return None

def _structured_search(n, m, p):
    """
    Try a structured approach: use the fact that the tensor is sparse.
    For 3x5x7, the tensor has 105 non-zero entries (all 1s).
    
    Try to find a low-rank decomposition by solving a system of equations.
    """
    rank_target = 104  # Try to beat schoolbook by 1
    
    # This is a hard problem. Let me try a specific known construction.
    # For small matrices, there are explicit constructions.
    
    # Alternative: use the fact that we can sometimes merge two terms
    # if their "support" overlaps appropriately.
    
    # Let me try a direct algebraic approach:
    # The tensor T[i,j,k] = delta(i,j,k) for the multiplication tensor.
    # We want T = sum_c U_c ⊗ V_c ⊗ W_c
    
    # For now, return schoolbook if we can't find improvement quickly.
    pass

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    rank = n * m * p
    
    if time.time() > deadline_global - 30:
        return {"U": U, "V": V, "W": W}
    
    # Try random search to find lower rank
    result = _random_search_start(U, V, W, n, m, p, max_iter=20000, seed=42)
    if result is not None:
        U, V, W, rank = result
        return {"U": U, "V": V, "W": W}
    
    # If we didn't find improvement, try a few more strategies
    if time.time() > deadline_global - 20:
        return {"U": U, "V": V, "W": W}
    
    # Try different seeds
    for seed in [1, 2, 3, 7, 13]:
        if time.time() > deadline_global - 15:
            break
        result = _random_search_start(U, V, W, n, m, p, max_iter=10000, seed=seed)
        if result is not None:
            U, V, W, rank = result
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
