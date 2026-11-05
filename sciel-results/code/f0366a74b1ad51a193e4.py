import time
from collections import defaultdict

def instance(n, p, seed):
    import random
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose(edges):
    if not edges:
        return []
    
    # Map vertex to set of neighbors via unused edges
    # We need to track which edges are used.
    # Since edges are undirected, we store them as sorted tuples.
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    paths = []
    
    while remaining:
        # Pick an arbitrary edge
        # To be deterministic and potentially better, pick an edge incident to a vertex with high degree?
        # Or just pick the first one from the set.
        # Sets are unordered in Python, but we can iterate.
        # To ensure reproducibility and avoid worst-case, we can sort or pick based on degree.
        # Let's pick an edge (u, v) such that u has the highest degree among remaining edges?
        # Simple approach: just pick any.
        
        # Find an edge
        e = next(iter(remaining))
        u, v = e
        
        # Remove edge
        remaining.remove(e)
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend from v
        curr = v
        while True:
            # Find a neighbor w of curr in adj[curr] that is not in path_set
            next_v = None
            for w in adj[curr]:
                if w not in path_set:
                    next_v = w
                    break
            
            if next_v is None:
                break
            
            # Use edge (curr, next_v)
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.append(next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                # This shouldn't happen if adj is consistent, but safety check
                break

        # Extend from u
        curr = u
        while True:
            next_v = None
            for w in adj[curr]:
                if w not in path_set:
                    next_v = w
                    break
            
            if next_v is None:
                break
            
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.insert(0, next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        paths.append(path)
        
    return paths

def construct():
    deadline = time.time() + 140
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline:
             # If we run out of time, we have to return what we have?
             # But the function must return 3 lists.
             # The loop is very fast, so this shouldn't trigger.
             pass
             
    return results
