"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _check_constraint(adj, n):
    """Check that no 3-edge path s-u-v-t has deg(s)==deg(t)."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if u < v:  # each edge once
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
    return True

def _havel_hakimi(degs):
    """Build adjacency list from degree sequence using Havel-Hakimi."""
    n = len(degs)
    # Create list of (degree, vertex)
    items = [(degs[i], i) for i in range(n)]
    adj = [set() for _ in range(n)]
    
    # We need to track remaining degrees
    rem = list(degs)
    for _ in range(n):
        # Find vertex with max remaining degree
        max_idx = -1
        max_val = -1
        for i in range(n):
            if rem[i] > max_val:
                max_val = rem[i]
                max_idx = i
        if max_val <= 0:
            break
        # Connect max_idx to next max_val vertices with highest remaining degrees
        # Exclude max_idx itself
        candidates = [(rem[j], j) for j in range(n) if j != max_idx and rem[j] > 0]
        candidates.sort(reverse=True)
        for k in range(max_val):
            if k >= len(candidates):
                return None  # Not graphical
            j = candidates[k][1]
            adj[max_idx].add(j)
            adj[j].add(max_idx)
            rem[j] -= 1
        rem[max_idx] = 0
    
    # Verify all degrees match
    for i in range(n):
        if len(adj[i]) != degs[i]:
            return None
    return adj

def construct():
    n = 15
    deadline = time.time() + 80.0
    
    # Try degree sequences with high edge count
    # Best case: 14,13,12,11,10,9,8,7,6,5,4,3,2,2,1 -> 52 edges
    base_degs = [14,13,12,11,10,9,8,7,6,5,4,3,2,2,1]
    
    # Try different permutations to find one that satisfies the constraint
    # The constraint violation only occurs if two degree-1 vertices are at distance 3
    # Let's try several orderings
    
    best_edges = None
    best_adj = None
    
    # Try the Havel-Hakimi construction with different tie-breaking
    # Actually, let's just try a few specific constructions
    
    # Strategy: place the two degree-1 vertices far apart
    # Vertex 0 and 1 have degree 1, connect them to high-degree vertices that aren't adjacent
    
    # Let me try: connect vertex 0 to vertex 14 (degree 14), vertex 1 to vertex 13 (degree 13)
    # Then vertex 14 is connected to everyone, vertex 13 to everyone except maybe vertex 14... 
    # Wait, degree 14 means connected to all 14 others.
    
    # Let me just use Havel-Hakimi and check
    adj = _havel_hakimi(base_degs)
    if adj is not None:
        if _check_constraint(adj, n):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return edges
    
    # If that fails, try modifying the degree sequence
    # Try: 14,13,12,11,10,9,8,7,6,5,4,3,2,1,1 (two 1s, no 2s... wait that's same sum)
    # Actually let me try different sequences
    
    # Try reducing some degrees slightly to get more flexibility
    for d2 in range(3, 0, -1):
        for d1 in range(d2, 0, -1):
            if time.time() > deadline:
                break
            degs = [14,13,12,11,10,9,8,7,6,5,4,3,d2,d1,1]
            if sum(degs) % 2 != 0:
                continue
            adj = _havel_hakimi(degs)
            if adj is not None and _check_constraint(adj, n):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                if best_edges is None or len(edges) > best_edges:
                    best_edges = len(edges)
                    best_adj = adj
    
    if best_adj is not None:
        edges = []
        for u in range(n):
            for v in best_adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    # Fallback: star graph
    return [[0, v] for v in range(1, n)]
# EVOLVE-BLOCK-END
