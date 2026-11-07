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
    Construct a valid solution using a greedy approach.
    """
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
        
    adj_masks = [0] * n
    for u, v in base_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    added = []
    
    # Strategy: Add long chords first.
    # Try adding edges (i, i+11), (i, i+10), etc.
    # We want to reduce diameter.
    
    candidates = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in base_set:
                continue
            candidates.append((i, j))
            
    # Sort by distance descending? Or just iterate.
    # Let's try a specific pattern that is known to work well:
    # Add (i, i+11) for all i.
    # Then add (i, i+5) for all i.
    # Then prune.
    
    # Let's just add a dense set of valid edges.
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in base_set:
                continue
            if _is_triangle_free_add_mask(adj_masks, i, j):
                # Add it
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
                added.append([i, j])
                
    # This adds ALL possible triangle-free edges.
    # This will definitely have diameter <= 4 (likely 2 or 3).
    # And it's a valid starting point for pruning.
    
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
    changed = True
    while changed and _time_ok():
        changed = False
        # Try to remove each edge
        # Iterate in reverse to handle indices safely
        for i in range(len(added) - 1, -1, -1):
            if not _time_ok():
                break
            # Remove edge i
            edge_to_remove = added[i]
            new_added = added[:i] + added[i+1:]
            
            if is_valid_added(new_added):
                added = new_added
                changed = True
    
    # 3. Optional: Local Search / Swapping
    # Try to replace an edge with another?
    # Usually pruning is enough.
    
    if _check_solution_fast(n, base, added):
        # Convert to list of lists of ints
        return [[int(u), int(v)] for u, v in added]
    
    # Fallback: Return empty if something went wrong (should not happen)
    return []

# EVOLVE-BLOCK-END
