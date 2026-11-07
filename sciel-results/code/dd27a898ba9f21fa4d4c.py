"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_biclique(adj, remaining_edges, anchor):
    """Find a maximal biclique anchored at `anchor` using remaining edges."""
    n = P['n']
    # Right side starts as neighbors of anchor that have remaining edges
    right = set()
    for v in adj[anchor]:
        if (min(anchor, v), max(anchor, v)) in remaining_edges:
            right.add(v)
    if not right:
        return None, None
    
    # Left side starts as {anchor}
    left = {anchor}
    
    # Iterate to find maximal biclique
    changed = True
    while changed:
        changed = False
        # Try to add vertices to left
        for v in list(adj[anchor]):
            if v in left:
                continue
            # v must connect to all in right
            if all(w in adj[v] for w in right):
                # And v must have at least one remaining edge to right? 
                # Actually, if v connects to all in right, and right is non-empty, 
                # the edges (v, w) are in the graph. 
                # We need them to be in remaining_edges to be covered by this biclique?
                # The biclique covers ALL edges in L x R.
                # If some (v, w) is NOT in remaining_edges, it means it was already covered.
                # So we cannot include v if it creates double coverage.
                # Thus, v can only be added if ALL (v, w) for w in right are in remaining_edges.
                if all((min(v, w), max(v, w)) in remaining_edges for w in right):
                    left.add(v)
                    changed = True
        
        # Try to add vertices to right
        for w in list(adj[anchor]):
            if w in right:
                continue
            # w must connect to all in left
            if all(u in adj[w] for u in left):
                # And all (u, w) for u in left must be in remaining_edges
                if all((min(u, w), max(u, w)) in remaining_edges for u in left):
                    right.add(w)
                    changed = True

    if not left or not right:
        return None, None
    
    return list(left), list(right)

def _greedy_partition(adj, order, remaining_edges=None):
    """Greedy partition following vertex order."""
    n = P['n']
    if remaining_edges is None:
        remaining = set()
        for u, v in P['edges']:
            remaining.add((min(u, v), max(u, v)))
    else:
        remaining = set(remaining_edges)
    
    bicliques = []
    
    for anchor in order:
        if not remaining:
            break
        
        # Find best biclique anchored at this vertex? 
        # No, the current logic picks the anchor from the order.
        # A better greedy: pick the anchor that yields the largest coverage.
        # But here we are forced to follow the order? 
        # Let's modify to pick the best anchor among all remaining vertices.
        pass
    
    # Revised Greedy:
    # At each step, pick the vertex u that, when used as anchor, yields the biclique covering the most remaining edges.
    # This is O(N * (N+E)) per step.
    
    bicliques = []
    remaining = set()
    for u, v in P['edges']:
        remaining.add((min(u, v), max(u, v)))
    
    while remaining:
        best_biclique = None
        best_count = -1
        best_anchor = -1
        
        for u in range(n):
            # Check if u has any remaining edges
            has_edge = False
            for v in adj[u]:
                if (min(u, v), max(u, v)) in remaining:
                    has_edge = True
                    break
            if not has_edge:
                continue
            
            L, R = _find_max_biclique(adj, remaining, u)
            if L is None:
                continue
            
            # Count remaining edges in L x R
            count = 0
            for u2 in L:
                for v2 in R:
                    e = (min(u2, v2), max(u2, v2))
                    if e in remaining:
                        count += 1
            
            if count > best_count:
                best_count = count
                best_biclique = (L, R)
                best_anchor = u
        
        if best_biclique is None:
            # Fallback: single edge
            e = next(iter(remaining))
            u, v = e
            bicliques.append([[u], [v]])
            remaining.discard(e)
        else:
            L, R = best_biclique
            bicliques.append([list(L), list(R)])
            for u2 in L:
                for v2 in R:
                    e = (min(u2, v2), max(u2, v2))
                    remaining.discard(e)
    
    return bicliques

def _validate(bicliques):
    """Check if biclique partition is valid."""
    n = P['n']
    adj = _build_adj()
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    for u, v in P['edges']:
        e = (min(u, v), max(u, v))
        if e not in covered:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    adj = _build_adj()
    n = P['n']
    
    best_result = None
    best_len = float('inf')
    
    # Run multiple random restarts
    for i in range(200):
        if time.time() > deadline - 5:
            break
        
        random.seed(i)
        order = list(range(n))
        random.shuffle(order)
        
        # The greedy algorithm above picks the best anchor at each step, 
        # so the order only matters for tie-breaking or if we modify it to use order.
        # Let's use a version that uses the order to pick the anchor from a subset?
        # No, the "best anchor" greedy is stronger.
        # To add randomness, we can slightly perturb the score or pick from top-k.
        
        result = _greedy_partition(adj, order)
        
        if len(result) < best_len:
            best_len = len(result)
            best_result = result
    
    if best_result is None:
        # Fallback
        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(n) if any(a == u for a, v in P['edges'])]
    
    if not _validate(best_result):
        return [[[u], [v for a, v in P['edges'] if a == u]] for u in range(n) if any(a == u for a, v in P['edges'])]
    
    return [[list(l), list(r)] for l, r in best_result]

# EVOLVE-BLOCK-END
