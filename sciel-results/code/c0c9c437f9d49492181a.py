"""Rewritten program: find rank-15 decomposition via grid partition."""
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    rank = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _solve_w_from_partition(n, m, p, grids):
    """Construct U, V, W from a partition into grids."""
    rank = len(grids)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c, grid in enumerate(grids):
        for (i, j, k) in grid:
            U[i*m + j][c] = 1
            V[j*p + k][c] = 1
            W[k*n + i][c] = 1
    return {"U": U, "V": V, "W": W}

def _find_partition(n, m, p, deadline):
    """Find a partition of all (i,j,k) triples into dyadic grids via backtracking."""
    all_triples = [(i,j,k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Generate all possible dyadic grids (size-2 Cartesian products)
    possible_grids = []
    # Type 1: {0,1} × {j} × {k}
    for j in range(m):
        for k in range(p):
            possible_grids.append(frozenset([(0,j,k), (1,j,k)]))
    # Type 2: {i} × {j1,j2} × {k}
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(j1+1, m):
                    possible_grids.append(frozenset([(i,j1,k), (i,j2,k)]))
    # Type 3: {i} × {j} × {k1,k2}
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(k1+1, p):
                    possible_grids.append(frozenset([(i,j,k1), (i,j,k2)]))
    
    # Map each triple to the list of grid indices containing it
    triple_to_grids = {t: [] for t in all_triples}
    for gi, grid in enumerate(possible_grids):
        for t in grid:
            triple_to_grids[t].append(gi)
    
    remaining = set(all_triples)
    grids_found = []
    
    def backtrack():
        if time.time() > deadline:
            return False
        if not remaining:
            return True
        # Pick the triple with the fewest available grids
        best_t = None
        best_count = float('inf')
        for t in remaining:
            count = 0
            for gi in triple_to_grids[t]:
                if possible_grids[gi].issubset(remaining):
                    count += 1
            if count < best_count:
                best_count = count
                best_t = t
            if count == 0:
                return False
        for gi in triple_to_grids[best_t]:
            grid = possible_grids[gi]
            if grid.issubset(remaining):
                remaining.difference_update(grid)
                grids_found.append(grid)
                if backtrack():
                    return True
                remaining.update(grid)
                grids_found.pop()
        return False
    
    if backtrack():
        return grids_found
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    grids = _find_partition(n, m, p, deadline)
    if grids is not None and len(grids) == 15:
        result = _solve_w_from_partition(n, m, p, grids)
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    return _schoolbook(n, m, p)
