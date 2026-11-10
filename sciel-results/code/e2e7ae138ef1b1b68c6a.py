"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _edges_to_adj(edge_set, n=18):
    adj = [0] * n
    for (u, v) in edge_set:
        if u == v:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free_adj(adj, n=18):
    """Fast K4-free check using bitmasks."""
    for u in range(n):
        au = adj[u]
        # Iterate over neighbors v of u with v > u
        v = u + 1
        while v < n:
            if (au >> v) & 1:
                common = au & adj[v]
                # Check if any edge exists within common (among vertices > v)
                cm = common & ~((1 << (v + 1)) - 1)
                while cm:
                    lsb = cm & (-cm)
                    w = lsb.bit_length() - 1
                    # Check if w has any neighbor in common other than itself
                    if adj[w] & (common & ~(1 << w)):
                        return False
                    cm ^= lsb
            v += 1
    return True

def _is_k4_free(edge_set, n=18):
    adj = _edges_to_adj(edge_set, n)
    return _is_k4_free_adj(adj, n)

def _max_triangle_free_subgraph_size_adj(adj, n=18, deadline=None):
    """
    Find the size of the largest triangle-free induced subgraph.
    Uses branch and bound on the minimum vertex cover of triangles.
    Returns the exact max triangle-free size if time permits, else a lower bound.
    """
    full = (1 << n) - 1
    
    # Precompute: for each vertex, the set of vertices that form a triangle with it
    # tri_with[u] = mask of vertices v such that u,v,w form a triangle for some w
    # Actually, let's precompute all triangles as a list of 3-bitmasks
    
    triangles = []
    for u in range(n):
        au = adj[u]
        v = u + 1
        while v < n:
            if (au >> v) & 1:
                common = au & adj[v] & ~((1 << (v + 1)) - 1)
                cm = common
                while cm:
                    lsb = cm & (-cm)
                    w = lsb.bit_length() - 1
                    triangles.append((1 << u) | (1 << v) | (1 << w))
                    cm ^= lsb
            v += 1
    
    if not triangles:
        return n
    
    # Greedy vertex cover: repeatedly remove vertex in most remaining triangles
    # This gives an upper bound on min vertex cover, hence lower bound on max TF size
    remaining_tris = list(triangles)
    removed = 0
    while remaining_tris:
        # Count triangles per vertex
        counts = [0] * n
        for tri in remaining_tris:
            for v in range(n):
                if (tri >> v) & 1:
                    counts[v] += 1
        # Pick vertex with max count
        best_v = -1
        best_c = 0
        for v in range(n):
            if (removed >> v) & 1:
                continue
            if counts[v] > best_c:
                best_c = counts[v]
                best_v = v
        if best_v == -1 or best_c == 0:
            break
        removed |= (1 << best_v)
        remaining_tris = [t for t in remaining_tris if not (t & (1 << best_v))]
    
    greedy_cover = bin(removed).count("1")
    lower_bound_tf = n - greedy_cover  # This is a LOWER bound on max TF size
    
    # Now use B&B to find if we can do better (smaller cover)
    # We want to find min vertex cover of triangles
    best_cover = greedy_cover
    
    # B&B: try to find a cover smaller than best_cover
    # State: set of uncovered triangles, current cover mask, cover size
    # Use memoization or just branch and bound with pruning
    
    deadline_check = deadline if deadline else time.time() + 1.0
    
    def _bnb(uncovered, cover_mask, cover_size, depth):
        nonlocal best_cover
        if time.time() > deadline_check:
            return
        if not uncovered:
            if cover_size < best_cover:
                best_cover = cover_size
            return
        if cover_size >= best_cover:
            return
        
        # Lower bound: need at least 1 more vertex
        # Pick a triangle with fewest available vertices to branch on
        tri = uncovered[0]
        # Available vertices in this triangle (not already in cover)
        avail = tri & ~cover_mask
        
        # Branch on each available vertex
        # Order by how many triangles it covers (for better pruning)
        vertices_in_tri = []
        m = avail
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            # Count how many uncovered triangles this vertex covers
            cnt = 0
            for t in uncovered:
                if (t >> v) & 1:
                    cnt += 1
            vertices_in_tri.append((cnt, v))
            m ^= lsb
        vertices_in_tri.sort(reverse=True)
        
        for cnt, v in vertices_in_tri:
            if time.time() > deadline_check:
                return
            new_cover = cover_mask | (1 << v)
            new_uncovered = [t for t in uncovered if not (t & (1 << v))]
            _bnb(new_uncovered, new_cover, cover_size + 1, depth + 1)
            if best_cover == 0:
                return
    
    _bnb(triangles, 0, 0, 0)
    
    return n - best_cover

def _max_triangle_free_subgraph_size(edge_set, n=18, deadline=None):
    adj = _edges_to_adj(edge_set, n)
    return _max_triangle_free_subgraph_size_adj(adj, n, deadline)

def _random_greedy_k4_free(n=18, seed=None, density_bias=0.0):
    """Build a random K4-free graph using greedy edge addition with optional density bias."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    adj = [0] * n
    edge_set = set()
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        # Check if adding (u,v) creates K4
        common = adj[u] & adj[v]
        ok = True
        cm = common
        while cm:
            lsb = cm & (-cm)
            w = lsb.bit_length() - 1
            if adj[w] & (common & ~(1 << w)):
                ok = False
                break
            cm ^= lsb
        if ok:
            # Optional: skip some edges based on density_bias
            if density_bias > 0 and rng.random() < density_bias:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
    
    return list(edge_set), adj

def _try_edge_swap(edge_set, adj, n=18, rng=None):
    """Try a single random edge swap and return (new_edge_set, new_adj, new_score) or None."""
    if not edge_set:
        return None
    
    rem_edge = rng.choice(list(edge_set))
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    cand = rng.choice(all_edges)
    
    if cand in edge_set:
        return None
    
    new_set = edge_set - {rem_edge} | {cand}
    new_adj = _edges_to_adj(new_set, n)
    
    if not _is_k4_free_adj(new_adj, n):
        return None
    
    return new_set, new_adj

def _local_search(edges, adj, n=18, max_iters=200, seed=None, deadline=None):
    """Local search to minimize max triangle-free subgraph size."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    current_score = _max_triangle_free_subgraph_size(edge_set, n, deadline)
    best_edges = list(edge_set)
    best_score = current_score
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        result = _try_edge_swap(edge_set, adj, n, rng)
        if result is None:
            continue
        
        new_set, new_adj = result
        new_score = _max_triangle_free_subgraph_size(new_set, n, deadline)
        
        if new_score < current_score:
            current_score = new_score
            edge_set = new_set
            adj = new_adj
            if current_score < best_score:
                best_score = current_score
                best_edges = list(edge_set)
    
    return best_edges, best_score

def _construct_balanced_tripartite(n=18):
    """Construct a balanced complete tripartite graph K6,6,6."""
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n = 18
    
    # Start with the known good construction: complete tripartite K6,6,6
    base_edges = _construct_balanced_tripartite(n)
    base_set = set(tuple(sorted(e)) for e in base_edges)
    base_adj = _edges_to_adj(base_set, n)
    base_score = _max_triangle_free_subgraph_size(base_set, n, deadline)
    
    best_edges = base_edges
    best_score = base_score
    
    # Try multiple random constructions + local search
    for trial in range(150):
        if time.time() > deadline:
            break
        
        # Build random K4-free graph with varying density
        density_bias = (trial % 10) * 0.05  # 0.0 to 0.45
        edges, adj = _random_greedy_k4_free(n, seed=trial, density_bias=density_bias)
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Check it's valid
        if not _is_k4_free_adj(adj, n):
            continue
        
        score = _max_triangle_free_subgraph_size(edge_set, n, deadline)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        # Local search
        ls_deadline = min(deadline, start_time + 147)
        ls_edges, ls_score = _local_search(edges, adj, n, max_iters=80, seed=trial, deadline=ls_deadline)
        
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
    
    # Final polish: try local search on the best found so far
    if time.time() < deadline - 5:
        best_set = set(tuple(sorted(e)) for e in best_edges)
        best_adj = _edges_to_adj(best_set, n)
        final_edges, final_score = _local_search(best_edges, best_adj, n, max_iters=100, seed=9999, deadline=deadline)
        if final_score < best_score:
            best_score = final_score
            best_edges = final_edges
    
    return [[u, v] for (u, v) in best_edges]
# EVOLVE-BLOCK-END
