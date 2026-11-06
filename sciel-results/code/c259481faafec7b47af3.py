"""C4-free graph construction for Ramsey number R(C4,K1,39)."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _has_c4_after_add(adj, u, v, n):
    """Check if adding edge (u,v) creates a C4.
    adj: list of int bitmasks.
    A C4 u-a-v-b-u exists iff |N(u) ∩ N(v)| >= 1 and there's a path of length 2 from u to v
    through two different vertices, AND u-v not already an edge.
    Actually, adding edge (u,v) creates a C4 iff there exist a,b with:
      a in N(u), b in N(v), a in N(v), b in N(u), a != b, a != v, b != u
    i.e., |N(u) ∩ N(v)| >= 2 (since u and v are not yet adjacent, N(u) and N(v) don't contain each other)
    Wait: if u-v is not an edge, then v not in N(u) and u not in N(v).
    A 4-cycle using new edge (u,v): u-a-v-b-u where a,b are common neighbors of u and v.
    So we need |N(u) ∩ N(v)| >= 2.
    """
    common = adj[u] & adj[v]
    # Count bits
    c = common.bit_count()
    return c >= 2

def _has_any_c4(adj, n):
    """Check if graph has any C4. For each pair of non-adjacent vertices, check common neighbors >= 2."""
    for u in range(n):
        for v in range(u+1, n):
            if not (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                if common.bit_count() >= 2:
                    return True
    return False

def _min_degree(adj, n):
    return min(adj[i].bit_count() for i in range(n))

def _construct_greedy(n, seed, deadline):
    """Try to build a C4-free graph on n vertices with min degree >= n-39."""
    rng = random.Random(seed)
    min_deg = n - 39
    if min_deg < 1:
        min_deg = 1
    
    # Start with empty graph
    adj = [0] * n
    
    # Generate candidate edges in random order
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            edges.append((u, v))
    rng.shuffle(edges)
    
    added = 0
    for (u, v) in edges:
        if time.time() > deadline:
            break
        # Check if adding (u,v) creates a C4
        if _has_c4_after_add(adj, u, v, n):
            continue
        # Add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        added += 1
    
    # Check min degree
    md = _min_degree(adj, n)
    if md < min_deg:
        # Try to add more edges to boost degree
        # Find vertices with low degree and try to connect them
        for _ in range(200):
            if time.time() > deadline:
                break
            low_verts = [i for i in range(n) if adj[i].bit_count() < min_deg]
            if not low_verts:
                break
            # Try all pairs of low-degree vertices or low with others
            improved = False
            for u in low_verts:
                for v in range(n):
                    if v == u or (adj[u] >> v) & 1:
                        continue
                    if not _has_c4_after_add(adj, u, v, n):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        md = _min_degree(adj, n)
        if md < min_deg:
            return None
    
    # Verify no C4
    if _has_any_c4(adj, n):
        return None
    
    # Build edge list
    edge_list = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                edge_list.append([u, v])
    
    return {'n': n, 'edges': edge_list}

def _construct_structured(n, seed, deadline):
    """Try a structured approach: use a polarity graph or similar."""
    # For C4-free graphs, the polarity graph of a finite projective plane is C4-free.
    # But let's try a different approach: random with repair.
    
    rng = random.Random(seed)
    min_deg = n - 39
    if min_deg < 1:
        min_deg = 1
    
    # Start with a random graph with appropriate density
    # Expected degree ~ min_deg, so probability p = min_deg / (n-1)
    p = min_deg / (n - 1)
    adj = [0] * n
    
    for u in range(n):
        for v in range(u+1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    # Remove edges that create C4s
    # Iterate: find a C4, remove one edge
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        # Find a C4
        found = False
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    if common.bit_count() >= 2:
                        # Found C4: u-a-v-b-u
                        # Remove edge (u, a) where a is a common neighbor
                        a = (common & -common).bit_length() - 1  # lowest set bit
                        # Remove edge (u, a)
                        adj[u] &= ~(1 << a)
                        adj[a] &= ~(1 << u)
                        found = True
                        break
            if found:
                break
        if not found:
            break
    
    # Check min degree
    md = _min_degree(adj, n)
    if md < min_deg:
        # Try to add edges to fix low-degree vertices
        for _ in range(300):
            if time.time() > deadline:
                break
            low_verts = [i for i in range(n) if adj[i].bit_count() < min_deg]
            if not low_verts:
                break
            improved = False
            for u in low_verts:
                candidates = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                rng.shuffle(candidates)
                for v in candidates:
                    if not _has_c4_after_add(adj, u, v, n):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        md = _min_degree(adj, n)
        if md < min_deg:
            return None
    
    # Final C4 check
    if _has_any_c4(adj, n):
        return None
    
    edge_list = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                edge_list.append([u, v])
    
    return {'n': n, 'edges': edge_list}

def _verify(result):
    """Verify the result is valid."""
    if result is None:
        return False
    n = result['n']
    edges = result['edges']
    if n < 1 or n > 47:
        return False
    # Build adjacency
    adj = [0] * n
    for (u, v) in edges:
        if u >= n or v >= n or u == v:
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    # Check min degree
    min_deg = n - 39
    for i in range(n):
        if adj[i].bit_count() < min_deg:
            return False
    # Check no C4
    if _has_any_c4(adj, n):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try n=47 first, then 46, then lower
    best_result = None
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline - 5:
            break
        
        # Try multiple strategies and seeds
        for strategy in ['greedy', 'structured']:
            for seed in range(20):
                if time.time() > deadline - 2:
                    break
                if strategy == 'greedy':
                    result = _construct_greedy(n, seed, deadline)
                else:
                    result = _construct_structured(n, seed, deadline)
                
                if result is not None and _verify(result):
                    if best_result is None or n > best_result['n']:
                        best_result = result
                        if n >= 46:
                            return result
                    break  # Found valid for this n, move to next n
        
        if best_result is not None and best_result['n'] >= 46:
            return best_result
    
    # Fallback: return the empty graph on 39 vertices
    if best_result is None:
        best_result = {'n': 39, 'edges': []}
    
    return best_result

# EVOLVE-BLOCK-END
