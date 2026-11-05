import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 135  # safety margin before 150s
    
    best_result = None
    best_n = 0
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple strategies
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            result = _construct_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                    return result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid small graph
    return {'n': 1, 'edges': []}


def _check_c4_add_fast(adj, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if C4 would be created.
    Optimized: uses bitset intersection to check for common neighbors efficiently."""
    # Adding (u,v) creates C4 if there exist x, y such that:
    # x is neighbor of u, y is neighbor of v, x is neighbor of y, and {x,y} != {u,v}
    # More precisely: there's a path u-x-y-v where x != v, y != u, x != y
    # This means: for some x in N(u)\{v}, x has a neighbor y in N(v)\{u}
    
    # Use numpy for fast bitset operations
    nu = adj[u]
    nv = adj[v]
    
    # For each neighbor x of u (excluding v), check if x shares any neighbor with v (excluding u)
    # We can vectorize: for all x in N(u), check if adj[x] intersects N(v)\{u}
    candidates = np.where(nu & ~nv)[0]  # neighbors of u not adjacent to v
    if len(candidates) == 0:
        return False
    
    # For each candidate x, check if adj[x] has any common neighbor with v (other than u)
    # adj[x] & nv gives common neighbors of x and v
    # We need to exclude u from this set
    for x in candidates:
        common = adj[x] & nv
        # Remove u from consideration
        if common[u]:
            common = common.copy()
            common[u] = False
        if common.any():
            # Also need x != v (already ensured) and the common neighbor y != x
            # y is in N(v) and N(x), y != u, y != v, y != x
            # Since x is in N(u) and we're checking common neighbors of x and v,
            # y is automatically != v (v not in N(v) for simple graph)
            # y != x because x not in N(x)
            # So just check if any common neighbor exists that isn't u
            return True
    
    return False


def _check_c4_add(adj, n, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if C4 would be created."""
    return _check_c4_add_fast(adj, u, v)


def _get_degree(adj, v):
    return int(adj[v].sum())


def _construct_c4_free(n, min_deg_req, deadline, seed_offset):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_req."""
    rng = random.Random(seed_offset * 12345 + 67890)
    
    # Strategy 1: Greedy with random restarts and priority queue
    adj = np.zeros((n, n), dtype=bool)
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Use a scoring function to prioritize edges
    def edge_score(u, v):
        du = int(adj[u].sum())
        dv = int(adj[v].sum())
        # Prefer edges that connect low-degree vertices
        return du + dv
    
    # Greedy construction with multiple passes
    for _pass in range(5):
        if time.time() > deadline:
            break
        
        # Sort edges by score (lower score = lower degree sum = more urgent)
        remaining = [(u, v) for (u, v) in all_edges if not adj[u, v]]
        rng.shuffle(remaining)
        remaining.sort(key=lambda e: edge_score(e[0], e[1]))
        
        improved = False
        for (u, v) in remaining:
            if time.time() > deadline:
                break
            if adj[u, v]:
                continue
            if not _check_c4_add(adj, n, u, v):
                adj[u, v] = True
                adj[v, u] = True
                improved = True
        
        if not improved:
            break
    
    # Check if we meet the requirement
    degrees = adj.sum(axis=1)
    if int(degrees.min()) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    # Strategy 2: Local search - try to fix low-degree vertices
    # Find vertices with degree < min_deg_req and try to add edges to them
    for _iter in range(200):
        if time.time() > deadline:
            break
        
        degrees = adj.sum(axis=1)
        low_verts = [v for v in range(n) if degrees[v] < min_deg_req]
        
        if not low_verts:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        improved = False
        for v in low_verts:
            if time.time() > deadline:
                break
            # Try adding edges from v to other vertices
            candidates = [u for u in range(n) if u != v and not adj[u, v]]
            rng.shuffle(candidates)
            for u in candidates:
                if not _check_c4_add(adj, n, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    improved = True
                    if degrees[v] >= min_deg_req:
                        break
        
        if not improved:
            break
    
    # Strategy 3: Edge swap local search
    # Try to swap edges to improve degrees while maintaining C4-freeness
    for _iter in range(300):
        if time.time() > deadline:
            break
        
        degrees = adj.sum(axis=1)
        if int(degrees.min()) >= min_deg_req:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find a low-degree vertex
        low_v = int(np.argmin(degrees))
        if degrees[low_v] >= min_deg_req:
            break
        
        # Try to add an edge to low_v by removing an edge from another vertex
        # First, try direct addition
        added = False
        for u in range(n):
            if u == low_v or adj[low_v, u]:
                continue
            if not _check_c4_add(adj, n, low_v, u):
                adj[low_v, u] = True
                adj[u, low_v] = True
                added = True
                break
        
        if added:
            continue
        
        # Try edge swap: remove edge (a,b) and add (low_v, c)
        # where removing (a,b) doesn't create issues and adding (low_v,c) is C4-free
        # This is complex, so try a simpler approach: random edge removal + addition
        if rng.random() < 0.3:  # Sometimes try swap
            # Pick a random existing edge to potentially remove
            existing_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
            if existing_edges:
                a, b = rng.choice(existing_edges)
                # Try removing (a,b) and adding (low_v, a) or (low_v, b)
                for new_v in [a, b]:
                    if new_v == low_v or adj[low_v, new_v]:
                        continue
                    # Check if we can add (low_v, new_v) after removing (a,b)
                    adj[a, b] = False
                    adj[b, a] = False
                    if not _check_c4_add(adj, n, low_v, new_v):
                        adj[low_v, new_v] = True
                        adj[new_v, low_v] = True
                        # Keep the edge removed if it helps
                        # Check if min degree improved or stayed same
                        new_degrees = adj.sum(axis=1)
                        if int(new_degrees.min()) >= int(degrees.min()):
                            degrees = new_degrees
                        else:
                            # Revert
                            adj[a, b] = True
                            adj[b, a] = True
                            adj[low_v, new_v] = False
                            adj[new_v, low_v] = False
                    else:
                        # Revert
                        adj[a, b] = True
                        adj[b, a] = True
    
    # Final check
    degrees = adj.sum(axis=1)
    if int(degrees.min()) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    return None


def _verify_c4_free(adj, n):
    """Verify graph is C4-free. Returns True if no C4 exists."""
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                continue
            # Check if there's a path u-x-y-v of length 3
            neighbors_u = set(np.where(adj[u])[0])
            neighbors_v = set(np.where(adj[v])[0])
            for x in neighbors_u:
                if x == v:
                    continue
                neighbors_x = set(np.where(adj[x])[0])
                common = neighbors_x & neighbors_v
                for y in common:
                    if y != u and y != v and y != x:
                        return False
    return True

# EVOLVE-BLOCK-END
