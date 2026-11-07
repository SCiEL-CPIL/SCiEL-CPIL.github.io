import time
from itertools import combinations
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = 24
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add(frozenset((u, v)))
    
    # Only even-odd edges not in path can be added
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and frozenset((u, v)) not in base_edges:
                candidates.append((u, v))
    
    deadline = time.time() + 85
    
    # Systematic search for minimum k
    for k in range(1, 12):
        if time.time() > deadline:
            break
        
        if k <= 2:
            # Exhaustive
            for combo in combinations(candidates, k):
                if time.time() > deadline:
                    break
                if _is_valid(combo, n, base_edges):
                    return [[int(u), int(v)] for u, v in combo]
        elif k == 3:
            # Exhaustive with early termination
            for combo in combinations(candidates, k):
                if time.time() > deadline:
                    break
                if _is_valid(combo, n, base_edges):
                    return [[int(u), int(v)] for u, v in combo]
        else:
            # For k >= 4, use randomized search with local improvement
            random.seed(123 + k)
            max_iters = 200000
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Random sample
                combo = tuple(random.sample(candidates, k))
                if _is_valid(combo, n, base_edges):
                    return [[int(u), int(v)] for u, v in combo]
                
                # Try local improvement: swap one edge
                if it % 10 == 0:  # occasionally do local search
                    combo_set = set(combo)
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        e_to_remove = random.choice(list(combo_set))
                        e_to_add = random.choice(candidates)
                        if e_to_add in combo_set:
                            continue
                        new_combo_set = combo_set - {e_to_remove} | {e_to_add}
                        new_combo = tuple(new_combo_set)
                        if _is_valid(new_combo, n, base_edges):
                            return [[int(u), int(v)] for u, v in sorted(new_combo_set)]
    
    # Fallback: construct a valid solution
    # Use a structured approach: add edges to create a "ladder" pattern
    # Vertices: 0-1-2-3-...-23
    # Add edges to create shortcuts while maintaining triangle-free
    
    # Strategy: create a bipartite structure with hubs
    # Even vertices: 0,2,4,...,22 (12 vertices)
    # Odd vertices: 1,3,5,...,23 (12 vertices)
    
    # Try a known good pattern
    solution = []
    # Connect pairs to reduce diameter
    # (0,23), (1,22), (2,21), ... but check triangle-free
    
    # Let's try: for i in 0..11, add edge (i, 23-i) if valid
    # But need to check triangle-free
    
    # Simpler: use a hub-and-spoke with two hubs
    # Hub 0 connects to odd vertices far away
    # Hub 23 connects to even vertices far away
    
    # Actually, let me just return a valid solution found by construction
    # The path has 24 vertices. To get diameter 4, we need shortcuts.
    
    # Construct: add edges (0,11), (1,12), (2,13), ..., (10,21), (11,22), (12,23)
    # This creates a "folded" structure
    
    # Let me verify a specific solution:
    # Add edges: (0, 12), (1, 13), (2, 14), (3, 15), (4, 16), (5, 17), (6, 18), (7, 19), (8, 20), (9, 21), (10, 22), (11, 23)
    # That's 12 edges. Let's try fewer.
    
    # Try: (0, 11), (1, 12), (2, 13), (3, 14), (4, 15), (5, 16), (6, 17), (7, 18), (8, 19), (9, 20), (10, 21), (11, 22)
    # 12 edges
    
    # Better: use a more efficient pattern
    # (0, 12), (1, 13), (2, 14), (3, 15), (4, 16), (5, 17), (6, 18), (7, 19), (8, 20), (9, 21), (10, 22), (11, 23)
    
    # Let me try a smaller set:
    # Key insight: we need to bridge the gap. The path is 23 edges long.
    # With diameter 4, we need the graph to be "dense" enough.
    
    # Try this pattern:
    fallback_edges = [
        (0, 12), (1, 13), (2, 14), (3, 15), (4, 16), (5, 17),
        (6, 18), (7, 19), (8, 20), (9, 21), (10, 22), (11, 23)
    ]
    
    if _is_valid(tuple(fallback_edges), n, base_edges):
        return [[int(u), int(v)] for u, v in fallback_edges]
    
    # Even simpler fallback: hub-based
    # Hub at 11 (center)
    hub = 11
    # Connect to vertices that are far
    targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22, 23]
    # But 11 is odd, so can only connect to even vertices
    # 11 is odd, so targets must be even
    even_targets = [0, 2, 4, 6, 8, 14, 16, 18, 20, 22]
    result = [(hub, t) for t in even_targets]
    
    if _is_valid(tuple(result), n, base_edges):
        return [[int(u), int(v)] for u, v in result]
    
    # Ultimate fallback: return a valid set
    # Just add enough edges to make it work
    all_added = []
    for i in range(0, 23, 2):
        all_added.append((i, i + 11))
    
    # Filter to only valid even-odd pairs
    valid_added = []
    for u, v in all_added:
        if (u % 2) != (v % 2) and frozenset((u, v)) not in base_edges:
            valid_added.append((u, v))
    
    if valid_added and _is_valid(tuple(valid_added), n, base_edges):
        return [[int(u), int(v)] for u, v in valid_added]
    
    # Last resort: return empty (will be invalid but better than crashing)
    return []

def _is_valid(added_edges, n, base_edges):
    """Check validity: triangle-free, connected, diameter <= 4."""
    all_edges = set(base_edges)
    for u, v in added_edges:
        all_edges.add(frozenset((u, v)))
    
    adj = [set() for _ in range(n)]
    for e in all_edges:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    
    # Triangle-free check
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            if adj[u] & adj[v]:
                return False
    
    # Diameter check
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= 4:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if any(d == -1 or d > 4 for d in dist):
            return False
    
    return True
