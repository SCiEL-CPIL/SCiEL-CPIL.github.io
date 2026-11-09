"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def _has_c4_in_color(adj, n):
    """Check if adjacency list for one color contains a C4."""
    # For each pair of vertices, check if they have >= 2 common neighbors
    for u in range(n):
        for v in range(u + 1, n):
            common = 0
            for w in adj[u]:
                if w in adj[v]:
                    common += 1
                    if common >= 2:
                        return True
    return False

def _validate_coloring(colors, n):
    """Validate that no color class contains a C4."""
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].append(j) if False else None
            idx += 1
    # Rebuild properly
    adj = [set() for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            idx += 1
    # Build neighbor lists per color
    neighbors = [defaultdict(set) for _ in range(5)]
    for c in range(5):
        for (u, v) in adj[c]:
            neighbors[c][u].add(v)
            neighbors[c][v].add(u)
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                if u in neighbors[c] and v in neighbors[c]:
                    common = neighbors[c][u] & neighbors[c][v]
                    if len(common) >= 2:
                        return False
    return True

def _lazebnik_woldar_26():
    """Lazebnik-Woldar construction for n=26 using F5 x F5."""
    n = 26
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            a1, b1 = divmod(i, 5)
            a2, b2 = divmod(j, 5)
            if a1 != a2:
                da = (a2 - a1) % 5
                db = (b2 - b1) % 5
                # slope = db * da^(-1) mod 5
                # da in {1,2,3,4}, inverses: 1->1, 2->3, 3->2, 4->4
                inv_da = [0, 1, 3, 2, 4][da]
                color = (db * inv_da) % 5
            else:
                # same column (a1 == a2), vertical edge
                color = 4
            colors.append(color)
    return n, colors

def _try_construction(n, colors):
    """Validate and return if valid."""
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    return _validate_coloring(colors, n)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Try the Lazebnik-Woldar n=26 construction
    n, colors = _lazebnik_woldar_26()
    if _try_construction(n, colors):
        return {'n': n, 'colors': colors}
    
    # Fallback: try n=11 cyclic construction (known valid)
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    if _try_construction(n, colors):
        return {'n': n, 'colors': colors}
    
    # Absolute fallback: n=1
    return {'n': 1, 'colors': []}
# EVOLVE-BLOCK-END
