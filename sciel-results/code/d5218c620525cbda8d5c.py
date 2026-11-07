"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 85.0  # safety margin before 90s kill

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs_distances(adj_masks, n, start):
    """
    adj_masks: list of ints, where adj_masks[i] is a bitmask of neighbors of i.
    Returns list of distances.
    """
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        mask = adj_masks[u]
        # Iterate through set bits
        m = mask
        while m:
            # Get lowest set bit
            lb = m & -m
            v = lb.bit_length() - 1
            m ^= lb
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _check_solution_fast(n, base_edges, added_edges):
    """
    Fast check using bitmasks.
    Returns True if valid.
    """
    # Build adjacency masks
    adj_masks = [0] * n
    
    def add_edge(u, v):
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    for u, v in base_edges:
        add_edge(u, v)
        
    for u, v in added_edges:
        # Check duplicate
        if (adj_masks[u] >> v) & 1:
            return False
        add_edge(u, v)
        
    # Triangle-free check
    # For each edge (u, v), adj[u] & adj[v] must be 0
    # Actually, just check for all pairs? No, just edges.
    # A triangle exists if there is an edge (u,v) such that adj[u] & adj[v] != 0.
    # Note: adj[u] contains v. adj[v] contains u.
    # So adj[u] & adj[v] will always have at least u and v?
    # No, adj[u] has v. adj[v] has u.
    # Intersection:
    # If there is a common neighbor w, then w is in adj[u] and adj[v].
    # So (adj[u] & adj[v]) will have bit w.
    # Also, does it have u? adj[v] has u. adj[u] does not have u. So no.
    # Does it have v? adj[u] has v. adj[v] does not have v. So no.
    # So (adj[u] & adj[v]) == 0 is the condition.
    
    # We need to iterate over edges to check this.
    # Or just check all pairs? 24*24/2 = 276 pairs.
    # Checking all pairs is easier and fast enough.
    for i in range(n):
        for j in range(i+1, n):
            if (adj_masks[i] >> j) & 1: # if edge exists
                if (adj_masks[i] & adj_masks[j]) != 0:
                    return False
                    
    # Connected + Diameter
    for start in range(n):
        dist = _bfs_distances(adj_masks, n, start)
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
            
    return True

def _get_diameter_fast(adj_masks, n):
    max_d = 0
    for start in range(n):
        dist = _bfs_distances(adj_masks, n, start)
        if -1 in dist:
            return -1
        d = max(dist)
        if d > max_d:
            max_d = d
    return max_d

def _is_triangle_free_add_mask(adj_masks, u, v):
    # Check if adding (u,v) creates a triangle
    # This is equivalent to checking if adj_masks[u] & adj_masks[v] is non-zero
    # BEFORE adding the edge?
    # If we add u-v, they become neighbors.
    # Triangle u-v-w exists if w is neighbor of u and w is neighbor of v.
    # So yes, check intersection of current neighborhoods.
    return (adj_masks[u] & adj_masks[v]) == 0

def _construct_initial(n, base_edges):
    """
    Construct a sparse valid solution.
    """
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    adj_masks = [0] * n
    for u, v in base_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    added = []
    
    # Strategy: Add specific long chords to reduce diameter quickly.
    # We try to add edges (i, i+12) for i in 0..11.
    # This creates a structure where any node is within 2 hops of the "middle".
    
    candidate_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in base_set:
                continue
            candidate_edges.append((i, j))
            
    # Prioritize edges that reduce diameter the most.
    # Heuristic: Long edges first.
    candidate_edges.sort(key=lambda e: e[1] - e[0], reverse=True)
    
    # Add edges one by one if they maintain triangle-free and improve/maintain diameter
    # Actually, just add a known good sparse set first.
    # Try adding (i, i+12) for i=0..11
    for i in range(12):
        j = i + 12
        if (i, j) in base_set: continue
        if _is_triangle_free_add_mask(adj_masks, i, j):
            adj_masks[i] |= (1 << j)
            adj_masks[j] |= (1 << i)
            added.append([i, j])
            
    # Check if this sparse set is valid (connected and diam <= 4)
    if _check_solution_fast(n, base_edges, added):
        return added
        
    # If not, fallback to adding more edges greedily
    # Re-initialize adj_masks for the greedy fill
    adj_masks = [0] * n
    for u, v in base_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    added = []
    
    for i, j in candidate_edges:
        if (i, j) in base_set: continue
        if _is_triangle_free_add_mask(adj_masks, i, j):
            adj_masks[i] |= (1 << j)
            adj_masks[j] |= (1 << i)
            added.append([i, j])
            
    return added

def construct():
    n = P['n']
    base = P['edges']
    
    # 1. Construct a dense valid solution
    added = _construct_initial(n, base)
    
    # 2. Prune edges
    # We want to remove as many as possible.
    # Iterate until no more edges can be removed.
    
    base_set = set()
    for u, v in base:
        base_set.add((min(u, v), max(u, v)))
        
    # Helper to check if a specific list of added edges is valid
    # We can do this by building masks from scratch each time, or updating.
    # Given n=24, rebuilding is fast.
    
    def is_valid_added(added_list):
        return _check_solution_fast(n, base, added_list)
        
    # Pruning Loop
    # We perform multiple passes to ensure we reach a local minimum.
    # Randomizing the order of checks can help escape local minima if we allow swaps, 
    # but for simple removal, order matters less if we do multiple passes.
    
    max_passes = 10
    for _ in range(max_passes):
        if not _time_ok():
            break
        changed = False
        # Shuffle to avoid bias? Or just iterate.
        # Let's iterate in reverse.
        for i in range(len(added) - 1, -1, -1):
            if not _time_ok():
                break
            edge_to_remove = added[i]
            new_added = added[:i] + added[i+1:]
            
            if is_valid_added(new_added):
                added = new_added
                changed = True
        if not changed:
            break
            
    # 3. Local Search: Try to swap edges to enable further removal?
    # If we are stuck, try to replace an edge e1 with e2 such that 
    # the new graph is valid. This doesn't reduce count immediately, 
    # but might allow removing e2 or another edge later.
    # Given time constraints, we stick to greedy removal which is fast.
    
    if _check_solution_fast(n, base, added):
        # Convert to list of lists of ints
        return [[int(u), int(v)] for u, v in added]
    
    # Fallback: Return empty if something went wrong (should not happen)
    return []

# EVOLVE-BLOCK-END
