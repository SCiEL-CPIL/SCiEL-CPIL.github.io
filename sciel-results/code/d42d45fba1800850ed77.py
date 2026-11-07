"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 16}


def _fast_diameter_check(n, adj):
    """
    Fast check: is graph connected and diameter <= 2?
    Uses bitmask BFS for speed.
    adj: list of int bitmasks, adj[i] has bit j set iff i~j.
    """
    allmask = (1 << n) - 1
    
    # Connectivity check via BFS from vertex 0
    visited = 1  # bit 0 set
    frontier = 1
    while frontier:
        # Compute neighbors of frontier
        nbrs = 0
        f = frontier
        while f:
            # get lowest set bit
            lb = f & (-f)
            idx = lb.bit_length() - 1
            nbrs |= adj[idx]
            f ^= lb
        nbrs &= ~visited
        visited |= nbrs
        frontier = nbrs
    
    if visited != allmask:
        return False  # disconnected
    
    # Check diameter <= 2: for every vertex, all others reachable within 2 steps
    # Equivalent: for every pair (u,v) not adjacent, there exists w such that u~w and w~v
    # Faster: for each vertex u, the set of vertices at distance exactly 2 from u
    # must cover all non-neighbors (and u itself is at distance 0)
    
    for u in range(n):
        # Neighbors of u
        nbrs_u = adj[u]
        # Vertices at distance <= 2 from u: nbrs_u | (neighbors of neighbors of u) | {u}
        # Compute neighbors of neighbors:
        nn = 0
        f = nbrs_u
        while f:
            lb = f & (-f)
            idx = lb.bit_length() - 1
            nn |= adj[idx]
            f ^= lb
        # Distance <= 2 set
        d2 = nbrs_u | nn | (1 << u)
        if d2 != allmask:
            return False  # some vertex is at distance > 2
    
    # Check diameter is exactly 2 (not 1): graph is not complete
    # If graph is complete, diameter is 1
    # But we also need to check it's not diameter 1
    # Actually, for our purposes, if diameter <= 2 and graph is not complete, diameter is exactly 2
    # Wait, we need diameter EXACTLY 2. So we need at least one pair at distance 2.
    # If the graph is complete, diameter is 1, which is invalid.
    # Check if graph is complete:
    for u in range(n):
        if adj[u] != (allmask ^ (1 << u)):
            return True  # Not complete, so diameter is exactly 2 (since we verified <= 2 and connected)
    
    return False  # Complete graph, diameter 1


def _is_critical(n, adj, u, v):
    """
    Check if removing edge (u,v) makes the graph disconnected or diameter > 2.
    adj: list of int bitmasks.
    u, v: endpoints of the edge to remove.
    Returns True if the edge is critical (removal breaks diameter <= 2 or connectivity).
    """
    # Temporarily remove edge
    adj[u] ^= (1 << v)
    adj[v] ^= (1 << u)
    
    critical = not _fast_diameter_check(n, adj)
    
    # Restore edge
    adj[u] ^= (1 << v)
    adj[v] ^= (1 << u)
    
    return critical


def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin within 60s limit
    
    n = P['n']
    half = n // 2
    
    # Start with K_{8,8}
    adj = [0] * n
    edges = []
    for u in range(half):
        for v in range(half, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([int(u), int(v)])
    
    # Verify base graph is valid (should be)
    if not _fast_diameter_check(n, adj):
        return edges  # Fallback, though this shouldn't happen
    
    # Check if base graph is fully critical
    # For K_{8,8}, every edge should be critical
    # Let's verify one to be sure
    # Actually, let's just proceed to try adding edges
    
    # Candidate edges to add: all non-edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                candidates.append((u, v))
    
    # We want to add as many edges as possible while maintaining:
    # 1. Diameter exactly 2
    # 2. Every edge is critical
    
    # Strategy: greedily try to add edges. After adding an edge, we must re-verify
    # that ALL existing edges are still critical, AND the new edge is critical.
    # This is expensive. Instead, we can use the insight that adding an edge
    # within A or within B creates a triangle, which might make some edges non-critical.
    
    # Let's try a more careful approach:
    # For each candidate edge, check if adding it preserves the property that
    # all edges (including the new one) are critical.
    
    # To speed this up, we can note that adding an edge (u,v) where u,v are in the same partition
    # creates a triangle with any common neighbor w. Then edge (u,w) might become non-critical
    # because u-v-w provides a path of length 2.
    
    # Actually, let's just do the full check for each candidate, but limit the number of attempts.
    
    # Sort candidates by some heuristic: prefer edges that might be "safer"
    # For now, just try them in order
    
    best_adj = [a for a in adj]
    best_edges = list(edges)
    
    for (u, v) in candidates:
        if time.time() > deadline:
            break
        
        # Try adding edge (u,v)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check if graph still has diameter exactly 2
        if not _fast_diameter_check(n, adj):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            continue
        
        # Now check if ALL edges are critical
        # This is the expensive part
        all_critical = True
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    if not _is_critical(n, adj, i, j):
                        all_critical = False
                        break
            if not all_critical:
                break
        
        if all_critical:
            # Keep the edge
            best_adj = [a for a in adj]
            best_edges.append([int(u), int(v)])
        else:
            # Remove the edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
    
    # Build final edge list from best_adj
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                final_edges.append([int(u), int(v)])
    
    return final_edges

# EVOLVE-BLOCK-END
