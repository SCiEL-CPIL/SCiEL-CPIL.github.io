import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145  # Leave 5s margin for 150s limit
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts with local search
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            result = _try_construct_fast(n, required_min_deg, deadline, attempt)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
        
        # If greedy fails for this n, try local search on best partial graph
        # For now, rely on restarts.
        
    # Fallback: return a valid smaller graph
    # For n=39, min degree >= 0, so empty graph works
    return {'n': 39, 'edges': []}

def _check_c4_add(adj, u, v):
    """
    Check if adding edge (u,v) creates a C4 in the graph defined by adj.
    A C4 is u-x-y-v-u.
    This exists if there is an edge (x,y) such that x is in N(u)\{v} and y is in N(v)\{u}.
    adj is a list of sets.
    """
    # N(u) and N(v) as sets
    nu = adj[u]
    nv = adj[v]
    
    # Check if there's an edge between N(u)\{v} and N(v)\{u}
    # Equivalent: exists x in N(u)\{v} such that N(x) intersects N(v)\{u}
    
    nv_set = nv.copy()
    nv_set.discard(u)
    
    for x in nu:
        if x == v:
            continue
        # Check if any neighbor of x is in nv_set
        for y in adj[x]:
            if y in nv_set:
                return True
    return False

def _try_construct_fast(n, required_min_deg, deadline, seed):
    """
    Greedy construction with efficient C4 checking using adjacency lists.
    """
    rng = np.random.RandomState(seed)
    
    # Use adjacency lists (list of sets) for fast neighbor lookup
    adj = [set() for _ in range(n)]
    degrees = [0] * n
    
    # Pre-compute all possible edges for random ordering
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    
    # First pass: try to add edges in random order to build a base graph
    for u, v in all_edges:
        if time.time() > deadline:
            return None
        if u in adj[v]:
            continue
        if _check_c4_add(adj, u, v):
            continue
        adj[u].add(v)
        adj[v].add(u)
        degrees[u] += 1
        degrees[v] += 1
    
    # Second pass: targeted edge addition for low-degree vertices
    for iteration in range(100):
        if time.time() > deadline:
            return None
        
        # Check if we are done
        if min(degrees) >= required_min_deg:
            break
        
        # Find vertices that need more edges
        needy = [i for i in range(n) if degrees[i] < required_min_deg]
        if not needy:
            break
        
        # Pick a random needy vertex
        u = needy[rng.randint(0, len(needy))]
        
        # Find candidates for u: non-neighbors
        candidates = [v for v in range(n) if v != u and v not in adj[u]]
        
        if not candidates:
            # u is connected to everyone, but degree < required? 
            # Try local search: remove an edge and re-add differently
            # For now, return None
            return None
        
        # Sort candidates by degree (ascending) to balance the graph
        candidates.sort(key=lambda v: degrees[v])
        
        added = False
        for v in candidates:
            if _check_c4_add(adj, u, v):
                continue
            # Safe to add
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            added = True
            break
        
        if not added:
            # Try a few other needy vertices
            success = False
            for _ in range(5):
                u2 = needy[rng.randint(0, len(needy))]
                candidates2 = [v for v in range(n) if v != u2 and v not in adj[u2]]
                if not candidates2:
                    continue
                candidates2.sort(key=lambda v: degrees[v])
                
                for v2 in candidates2:
                    if not _check_c4_add(adj, u2, v2):
                        adj[u2].add(v2)
                        adj[v2].add(u2)
                        degrees[u2] += 1
                        degrees[v2] += 1
                        success = True
                        break
                if success:
                    break
            
            if not success:
                # Stuck - try local search: remove a random edge and try to re-add
                # Find an edge to remove
                if not all_edges:
                    return None
                # Pick a random existing edge to remove
                existing_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            existing_edges.append((i, j))
                
                if not existing_edges:
                    return None
                
                rem_u, rem_v = existing_edges[rng.randint(0, len(existing_edges))]
                # Remove the edge
                adj[rem_u].discard(rem_v)
                adj[rem_v].discard(rem_u)
                degrees[rem_u] -= 1
                degrees[rem_v] -= 1
                
                # Now try to add edges to needy vertices
                needy = [i for i in range(n) if degrees[i] < required_min_deg]
                if not needy:
                    # We removed an edge but all degrees are still ok? That's fine
                    continue
                
                success = False
                for _ in range(20):
                    u3 = needy[rng.randint(0, len(needy))]
                    candidates3 = [v for v in range(n) if v != u3 and v not in adj[u3]]
                    if not candidates3:
                        continue
                    candidates3.sort(key=lambda v: degrees[v])
                    
                    for v3 in candidates3:
                        if not _check_c4_add(adj, u3, v3):
                            adj[u3].add(v3)
                            adj[v3].add(u3)
                            degrees[u3] += 1
                            degrees[v3] += 1
                            success = True
                            break
                    if success:
                        break
                
                if not success:
                    return None

    # Final check
    if min(degrees) < required_min_deg:
        return None

    # Extract edges
    edges = []
    for u in range(n):
        for v in sorted(adj[u]):
            if u < v:
                edges.append([u, v])
                
    return (n, edges)

# EVOLVE-BLOCK-END
