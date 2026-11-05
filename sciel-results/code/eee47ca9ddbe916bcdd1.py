import time
import random
import itertools

_START = time.time()
_DEADLINE = _START + 115.0

def _deadline():
    return time.time() >= _DEADLINE

def _is_regular_mask(adj, mask):
    """Check if induced subgraph on mask is regular."""
    first = -1
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        m ^= lsb
        deg = (adj[v] & mask).bit_count()
        if first == -1:
            first = deg
        elif deg != first:
            return False
    return True

def _find_one_regular_subgraph(adj, n):
    """Find one induced regular subgraph of size >= 6. Returns mask or 0."""
    # Check smaller sizes first as they are more likely to exist and cheaper to find?
    # Actually, checking all is necessary.
    # Optimization: If we find one, return immediately.
    for k in range(6, n + 1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if _is_regular_mask(adj, mask):
                return mask
    return 0

def _edges_in_mask(mask):
    """Yield edges (u, v) where u < v and both in mask."""
    verts = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            u, v = verts[i], verts[j]
            yield u, v

def _repair_graph(adj, n, rng):
    """
    Iteratively find a regular subgraph and break it by flipping a random edge within it.
    """
    current_adj = adj[:]
    # Limit iterations to avoid infinite loops if stuck in a cycle
    # 5000 iterations should be enough for n=22
    for _ in range(5000):
        if _deadline():
            break
        
        mask = _find_one_regular_subgraph(current_adj, n)
        if mask == 0:
            return current_adj # Valid graph found
        
        # Pick a random edge within the regular subgraph to flip
        # Flipping an internal edge definitely breaks the regularity of that specific subgraph
        # (degrees of two vertices change by +/-1)
        edges = list(_edges_in_mask(mask))
        if not edges:
            # Should not happen for k>=6
            break
        u, v = rng.choice(edges)
        
        # Flip edge (u, v)
        current_adj[u] ^= (1 << v)
        current_adj[v] ^= (1 << u)
        
    return current_adj

def _valid(adj, n):
    return _find_one_regular_subgraph(adj, n) == 0

def _random_graph(n, p, rng):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    rng = random.Random(42)
    
    # Try from largest n down
    for n in range(22, 19, -1):
        if _deadline():
            break
            
        found = False
        # Try multiple random starts
        for attempt in range(200):
            if _deadline():
                break
                
            # Random density, biased towards 0.5
            p = rng.uniform(0.3, 0.7)
            adj = _random_graph(n, p, rng)
            
            # Repair
            repaired_adj = _repair_graph(adj, n, rng)
            
            # Final check
            if _valid(repaired_adj, n):
                found = True
                return {'n': n, 'edges': _edges_from_adj(repaired_adj, n)}
        
        # If not found, try a slightly different strategy?
        # Maybe start with a known valid smaller graph and extend?
        # But random repair is usually strong enough for n=20-22.
        
    # Fallback to n=20 if we ran out of time or failed
    # Try to find a valid graph on n=20 one last time
    n = 20
    for attempt in range(100):
        if _deadline():
            break
        p = 0.5
        adj = _random_graph(n, p, rng)
        repaired_adj = _repair_graph(adj, n, rng)
        if _valid(repaired_adj, n):
            return {'n': n, 'edges': _edges_from_adj(repaired_adj, n)}
            
    # Absolute fallback: return a small valid graph (e.g., n=5)
    # A path graph P5 has no regular induced subgraph on 6 vertices (trivially)
    # But we need to maximize n.
    # If we fail to find 20, return whatever we have or a trivial one.
    # Let's return n=5 to be safe, though score will be low.
    adj = [0]*5
    # Make it a path 0-1-2-3-4
    adj[0] = 1 << 1
    adj[1] = (1 << 0) | (1 << 2)
    adj[2] = (1 << 1) | (1 << 3)
    adj[3] = (1 << 2) | (1 << 4)
    adj[4] = 1 << 3
    return {'n': 5, 'edges': _edges_from_adj(adj, 5)}
