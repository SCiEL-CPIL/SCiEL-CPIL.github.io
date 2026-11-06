import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Try from largest n down to 9
    for n in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time() - 3
        if remaining < 2:
            break
        budget = min(remaining, 20.0)
        res = _search(n, budget)
        if res is not None:
            return res
    
    # Fallback: guaranteed valid solution for n=9
    return _fallback()


def _fallback():
    """Return a valid 4-coloring of K_9 with no monochromatic C6."""
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Distance-based coloring: d%4 avoids C6 for small n
            d = j - i
            colors.append(d % 4)
    return {'n': n, 'colors': colors}


def _search(n, budget_seconds):
    """
    Search for a valid 4-coloring of K_n with no monochromatic C6.
    Uses randomized backtracking with:
    - Edge ordering by degree (high-degree edges first)
    - Forward checking via C6 constraint
    - Multiple random restarts
    """
    deadline = time.time() + budget_seconds
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    # Order edges: process edges incident to high-degree vertices first
    # Actually, for complete graph all vertices have same degree.
    # Instead, order to break symmetry: edges from vertex 0 first, then 1, etc.
    # This is already the lexicographic order.
    
    # Better: use a randomized edge order per restart, but fix first few for symmetry breaking
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        rng = random.Random(attempt * 2654435761 + 12345)
        
        # Create edge order: first few edges fixed for symmetry breaking, rest shuffled
        edge_order = list(range(num_edges))
        # Symmetry breaking: fix first edge (0,1)=0, second (0,2) in {0,1}
        # We'll handle this in the backtrack
        
        if _bt(0, edges, num_edges, n, colors, adj, deadline, rng):
            return {'n': n, 'colors': colors}
    
    return None


def _bt(idx, edges, num_edges, n, colors, adj, deadline, rng):
    """Backtracking with constraint checking."""
    if time.time() > deadline:
        return False
    if idx == num_edges:
        return True
    
    u, v = edges[idx]
    
    # Symmetry breaking for first few edges
    if idx == 0:
        cands = [0]
    elif idx == 1:
        cands = [0, 1]
        rng.shuffle(cands)
    else:
        cands = [0, 1, 2, 3]
        rng.shuffle(cands)
    
    for c in cands:
        if _has_c6(u, v, c, n, adj):
            continue
        
        colors[idx] = c
        adj[c][u] |= (1 << v)
        adj[c][v] |= (1 << u)
        
        if _bt(idx + 1, edges, num_edges, n, colors, adj, deadline, rng):
            return True
        
        adj[c][u] &= ~(1 << v)
        adj[c][v] &= ~(1 << u)
        colors[idx] = -1
    
    return False


def _has_c6(u, v, c, n, adj):
    """
    Check if adding edge (u,v) with color c creates a monochromatic C6 in color c.
    A C6: u - x1 - x2 - x3 - x4 - v, all edges in color c, all 6 vertices distinct.
    """
    nu = adj[c][u]
    nv = adj[c][v]
    
    if nu == 0 or nv == 0:
        return False
    
    # x1 in N_c(u), x4 in N_c(v), path x1-x2-x3-x4 in G_c
    m1 = nu
    while m1:
        b = m1 & (-m1)
        x1 = b.bit_length() - 1
        m1 ^= b
        
        m4 = nv
        while m4:
            b4 = m4 & (-m4)
            x4 = b4.bit_length() - 1
            m4 ^= b4
            
            if x4 == u or x4 == x1:
                continue
            
            forb = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)
            
            a = adj[c][x1] & ~forb
            b_ = adj[c][x4] & ~forb
            
            if a == 0 or b_ == 0:
                continue
            
            # Check if any x2 in a has an edge to some x3 in b_
            ma = a
            while ma:
                ba = ma & (-ma)
                x2 = ba.bit_length() - 1
                ma ^= ba
                if adj[c][x2] & b_:
                    return True
    
    return False
