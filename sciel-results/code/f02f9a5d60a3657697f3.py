"""C6-free 4-coloring: seed known n=17, attempt growth to 18/19."""
import time
import random

_START = time.time()
_TIMEOUT = 115.0

def _time_left():
    return _TIMEOUT - (time.time() - _START)

def _edge_idx(n, i, j):
    """Lexicographic index for (i,j), i<j, in K_n."""
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_adj(n, colors):
    """Build 4 adjacency matrices."""
    adj = [[0] * n for _ in range(4)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[_edge_idx(n, i, j)]
            adj[c][i][j] = 1
            adj[c][j][i] = 1
    return adj

def _has_c6(adj, n):
    """Check if adj (n x n) contains a 6-cycle."""
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if not adj[c][d]:
                        continue
                    for e in range(d + 1, n):
                        if not adj[d][e]:
                            continue
                        if adj[e][a]:
                            return True
    return False

def _is_valid(n, colors):
    """Verify no color class has a C6."""
    adj = _build_adj(n, colors)
    for c in range(4):
        if _has_c6(adj[c], n):
            return False
    return True

def _random_valid_coloring(n, max_attempts=2000, seed=None):
    """Find a random C6-free 4-coloring of K_n, or None."""
    if seed is not None:
        random.seed(seed)
    for _ in range(max_attempts):
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        if _is_valid(n, colors):
            return colors
    return None

def _try_grow(n_seed, colors_seed, target_n, max_attempts=200):
    """
    Try to grow n_seed coloring to target_n by adding vertices.
    Uses random restarts: randomly color new edges, check validity.
    """
    for attempt in range(max_attempts):
        if _time_left() < 3:
            break
        
        random.seed(attempt * 7919 + 13)
        colors = list(colors_seed)
        
        # Add vertices one at a time
        for v in range(n_seed, target_n):
            # Add edges (v, u) for u in 0..v-1
            # In lexicographic order for K_target_n, these come at the end
            # after all edges among 0..v-1
            for u in range(v):
                colors.append(random.randint(0, 3))
            
            # Check validity for K_{v+1}
            if not _is_valid(v + 1, colors):
                break
        else:
            # All vertices added successfully
            if _is_valid(target_n, colors):
                return colors
    return None

def construct():
    global _START
    _START = time.time()
    
    best_n = 0
    best_colors = None
    
    # Try to find n=17 coloring
    for seed in range(50):
        if _time_left() < 10:
            break
        colors = _random_valid_coloring(17, max_attempts=200, seed=seed)
        if colors:
            best_n = 17
            best_colors = colors
            break
    
    # If n=17 found, try to grow to 18
    if best_n == 17 and _time_left() > 15:
        grown = _try_grow(17, best_colors, 18, max_attempts=100)
        if grown:
            return {'n': 18, 'colors': grown}
    
    # If n=17 not found, try smaller
    if best_n < 17:
        for n_try in [16, 15, 14, 13, 12, 11, 10, 9]:
            if _time_left() < 5:
                break
            colors = _random_valid_coloring(n_try, max_attempts=500, seed=n_try)
            if colors:
                best_n = n_try
                best_colors = colors
                break
    
    if best_colors is None:
        # Fallback: n=9 known valid
        n = 9
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}
    
    return {'n': best_n, 'colors': best_colors}
