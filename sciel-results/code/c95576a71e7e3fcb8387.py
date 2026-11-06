import time
import random
import numpy as np

def _build_adj(edges, n):
    """Build adjacency matrix from edge list."""
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _is_k4_free(adj, n):
    """Check if graph is K4-free."""
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                continue
            for k in range(j + 1, n):
                if not adj[i][k] or not adj[j][k]:
                    continue
                for l in range(k + 1, n):
                    if adj[i][l] and adj[j][l] and adj[k][l]:
                        return False
    return True

def _get_triangles(adj, n):
    """Get all triangles as bitmasks."""
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                continue
            for k in range(j + 1, n):
                if adj[i][k] and adj[j][k]:
                    triangles.append((1 << i) | (1 << j) | (1 << k))
    return triangles

def _max_triangle_free_induced(adj, n):
    """
    Find the size of the largest triangle-free induced subgraph.
    Uses branch and bound with bit-parallel operations.
    """
    triangles = _get_triangles(adj, n)
    
    if not triangles:
        return n
    
    # For n=18, we can use a clever approach:
    # The maximum triangle-free set is the maximum independent set in the 
    # "triangle conflict graph" - but that's complex.
    
    # Better: Use the fact that we want min vertex cover of triangles.
    # For n=18, we can do a smart B&B.
    
    # Precompute: for each vertex, which triangles it belongs to
    vertex_to_triangles = [[] for _ in range(n)]
    for idx, t in enumerate(triangles):
        for v in range(n):
            if (t >> v) & 1:
                vertex_to_triangles[v].append(idx)
    
    # Use a recursive B&B for minimum vertex cover
    # State: set of remaining triangles (as indices), current cover size
    # We use memoization on the frozenset of remaining triangle indices
    
    # For speed, use a greedy upper bound first
    def greedy_vertex_cover():
        remaining = set(range(len(triangles)))
        cover = 0
        while remaining:
            # Count how many remaining triangles each vertex hits
            counts = [0] * n
            for tidx in remaining:
                t = triangles[tidx]
                for v in range(n):
                    if (t >> v) & 1:
                        counts[v] += 1
            # Pick vertex with max count
            best_v = max(range(n), key=lambda v: counts[v])
            if counts[best_v] == 0:
                break
            cover += 1
            # Remove all triangles containing best_v
            new_remaining = set()
            for tidx in remaining:
                t = triangles[tidx]
                if not ((t >> best_v) & 1):
                    new_remaining.add(tidx)
            remaining = new_remaining
        return cover
    
    ub = greedy_vertex_cover()
    
    # Now do B&B to find exact minimum vertex cover
    # We'll use a recursive approach with pruning
    
    def bnb_vertex_cover(remaining_triangles, current_depth, best_so_far):
        if not remaining_triangles:
            return 0
        if current_depth >= best_so_far:
            return float('inf')
        
        # Lower bound: at least 1 if there are triangles
        if current_depth + 1 >= best_so_far:
            return float('inf')
        
        # Pick the triangle with the fewest "available" vertices
        # (vertices that can still be added to cover)
        min_tri = None
        min_options = float('inf')
        for tidx in remaining_triangles:
            t = triangles[tidx]
            options = 0
            for v in range(n):
                if (t >> v) & 1:
                    options += 1
            if options < min_options:
                min_options = options
                min_tri = t
            if options == 0:
                return float('inf')
        
        if min_options == 0:
            return float('inf')
        
        best = float('inf')
        t = min_tri
        for v in range(n):
            if (t >> v) & 1:
                # Try adding v to cover
                new_remaining = set()
                for tidx in remaining_triangles:
                    t2 = triangles[tidx]
                    if not ((t2 >> v) & 1):
                        new_remaining.add(tidx)
                
                sub_cost = bnb_vertex_cover(new_remaining, current_depth + 1, best_so_far)
                if sub_cost + 1 < best:
                    best = sub_cost + 1
                    if best == min_options:  # Can't do better
                        break
        
        return best
    
    # Run B&B
    min_cover = bnb_vertex_cover(set(range(len(triangles))), 0, ub)
    
    # The answer is n - min_cover
    return n - min_cover

def _get_paley_edges(q):
    """Get edges for Paley graph of order q."""
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    edges = []
    for u in range(q):
        for v in range(u + 1, q):
            diff = (v - u) % q
            if diff in residues:
                edges.append([u, v])
    return edges

def _get_random_regular_edges(n, d, seed=42):
    """Generate a random d-regular graph on n vertices."""
    rng = random.Random(seed)
    edges = set()
    
    # Use configuration model
    half_edges = [v for v in range(n) for _ in range(d)]
    rng.shuffle(half_edges)
    
    adj = [[False] * n for _ in range(n)]
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i + 1]
        if u != v and not adj[u][v]:
            adj[u][v] = True
            adj[v][u] = True
            edges.add((min(u, v), max(u, v)))
    
    # Convert to edge list
    return [[u, v] for u, v in sorted(edges)]

def _get_blowup_edges(parts, base_edges):
    """Create blow-up of base graph with given part sizes."""
    edges = []
    n = sum(parts)
    # Map vertex indices
    vertex_map = []
    for i, size in enumerate(parts):
        for _ in range(size):
            vertex_map.append(i)
    
    for u, v in base_edges:
        if u >= len(parts) or v >= len(parts):
            continue
        for i in range(parts[u]):
            for j in range(parts[v]):
                # Find actual vertex indices
                # This is simplified - just use the mapping
                pass
    
    # Simpler approach: just create complete multipartite
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 18
    
    best_edges = None
    best_score = float('inf')
    
    def try_candidate(edges):
        nonlocal best_edges, best_score
        if time.time() > deadline:
            return
        adj = _build_adj(edges, n)
        if not _is_k4_free(adj, n):
            return
        score = _max_triangle_free_induced(adj, n)
        if score < best_score:
            best_score = score
            best_edges = edges[:]
    
    # Candidate 1: Paley graph P(17) + isolated vertex
    edges_17 = _get_paley_edges(17)
    try_candidate(edges_17)
    
    # Candidate 2: Two disjoint Paley graphs P(9)
    edges_9_1 = _get_paley_edges(9)
    edges_9_2 = [[u + 9, v + 9] for u, v in edges_9_1]
    edges_18 = edges_9_1 + edges_9_2
    try_candidate(edges_18)
    
    # Candidate 3: K6,6,6 (Baseline)
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    edges_k666 = [[a, b] for i in range(3) for j in range(i + 1, 3) 
                  for a in parts[i] for b in parts[j]]
    try_candidate(edges_k666)
    
    # Candidate 4: Various random regular graphs
    for d in [5, 6, 7, 8, 9]:
        if time.time() > deadline:
            break
        for seed in range(5):
            edges_rr = _get_random_regular_edges(n, d, seed=seed * 100 + d)
            try_candidate(edges_rr)
    
    # Candidate 5: Local search refinement
    if best_edges is not None and time.time() < deadline:
        current_edges = best_edges[:]
        current_adj = _build_adj(current_edges, n)
        current_score = best_score
        
        # Build list of non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not current_adj[u][v]:
                    non_edges.append((u, v))
        
        random.shuffle(non_edges)
        
        improved = True
        max_iter = 0
        while improved and time.time() < deadline and max_iter < 50:
            improved = False
            max_iter += 1
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                
                # Check if adding edge (u,v) creates a K4
                creates_k4 = False
                common = []
                for w in range(n):
                    if w != u and w != v and current_adj[u][w] and current_adj[v][w]:
                        common.append(w)
                
                for i in range(len(common)):
                    for j in range(i + 1, len(common)):
                        if current_adj[common[i]][common[j]]:
                            creates_k4 = True
                            break
                    if creates_k4:
                        break
                
                if not creates_k4:
                    # Try adding this edge
                    current_adj[u][v] = True
                    current_adj[v][u] = True
                    current_edges.append([u, v])
                    
                    new_score = _max_triangle_free_induced(current_adj, n)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # Rebuild non_edges list
                        non_edges = []
                        for uu in range(n):
                            for vv in range(uu + 1, n):
                                if not current_adj[uu][vv]:
                                    non_edges.append((uu, vv))
                        random.shuffle(non_edges)
                        break
                    else:
                        # Revert
                        current_adj[u][v] = False
                        current_adj[v][u] = False
                        current_edges.pop()
            
            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges[:]
        
        if current_score < best_score:
            best_score = current_score
            best_edges = current_edges[:]
    
    if best_edges is None:
        # Fallback to K6,6,6
        return edges_k666
    
    return best_edges
