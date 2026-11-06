"""Triangle-free 4-chromatic graphs on 16 vertices via core extension and local search."""
# EVOLVE-BLOCK-START
import random
import time

def _build_grotzsch_edges():
    """
    Return the edge list of the Grötzsch graph (Mycielski of C5) on vertices 0..10.
    Vertices 11..15 are isolated in the initial state.
    """
    # C5: 0-1-2-3-4-0
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    
    # Mycielski construction:
    # v0=0, v1=1, v2=2, v3=3, v4=4
    # w0=5, w1=6, w2=7, w3=8, w4=9
    # u=10
    
    for i in range(5):
        w_idx = 5 + i
        prev = (i - 1) % 5
        next = (i + 1) % 5
        edges.append([w_idx, prev])
        edges.append([w_idx, next])
        
    for i in range(5):
        edges.append([10, 5 + i])
        
    return edges

def _make_bitmask_adj(edge_list, n=16):
    """Convert edge list to list of integer bitmasks representing adjacency."""
    adj = [0] * n
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _greedy_fill(adj, n=16):
    """
    Greedily add all possible edges that do not create triangles.
    Returns updated adj and the number of edges added.
    """
    added = 0
    changed = True
    # We iterate until no more edges can be added
    while changed:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    # Check if adding (u, v) creates a triangle
                    if adj[u] & adj[v] == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added += 1
                        changed = True
    return adj, added

def _count_edges(adj, n=16):
    count = 0
    for u in range(n):
        count += bin(adj[u]).count('1')
    return count // 2

def _local_search(seed, deadline):
    n = 16
    rng = random.Random(seed)
    
    # Initialize with Grotzsch
    grotzsch = _build_grotzsch_edges()
    adj = _make_bitmask_adj(grotzsch, n)
    
    # Initial greedy fill
    adj, _ = _greedy_fill(adj, n)
    current_count = _count_edges(adj, n)
    best_count = current_count
    best_adj = adj[:]
    
    # Local Search Loop
    while time.time() < deadline:
        # Try a few different move types
        move_type = rng.choice([1, 1, 2, 3]) # 1: remove 1, 2: remove 2, 3: remove 3
        
        saved_adj = adj[:]
        
        # Remove random edges
        edges_to_remove = []
        # Get list of current edges
        current_edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if u < v:
                    current_edges.append((u, v))
                mask ^= lb
        
        if len(current_edges) < move_type:
            break
            
        edges_to_remove = rng.sample(current_edges, move_type)
        
        for u, v in edges_to_remove:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        # Greedy refill
        adj, _ = _greedy_fill(adj, n)
        new_count = _count_edges(adj, n)
        
        if new_count > best_count:
            best_count = new_count
            best_adj = adj[:]
        elif new_count == best_count:
            # Keep current state to explore neighborhood (optional, but helps escaping)
            # Actually, reverting is safer if we want to stay on the "ridge"
            # But if we are stuck, moving to an equal state might help.
            # Let's keep the new state if it's equal, to allow drift.
            pass
        else:
            # Revert
            adj = saved_adj
            
    # Convert best_adj to edge list
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result_edges.append([u, v])
                
    return result_edges, len(result_edges)

def construct():
    """
    Construct a triangle-free graph on 16 vertices that is not 3-colorable,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Run multiple instances of local search with different seeds
    seed = 0
    while time.time() < deadline - 5.0:
        edges, count = _local_search(seed, deadline)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        # If we hit a very high count, stop?
        # Max is 64.
        if best_count >= 60:
            break

    # Fallback
    if not best_edges:
        best_edges = _build_grotzsch_edges()
        
    return best_edges
# EVOLVE-BLOCK-END
