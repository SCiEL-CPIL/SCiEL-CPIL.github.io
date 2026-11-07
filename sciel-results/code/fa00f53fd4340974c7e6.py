"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_biclique(remaining_edges, adj, n):
    """Find a large biclique in the remaining edges using a heuristic."""
    if not remaining_edges:
        return None, None
    
    edge_set = set((min(u,v), max(u,v)) for u, v in remaining_edges)
    
    # Try multiple strategies: pick vertices by degree in remaining graph
    remaining_adj = [set() for _ in range(n)]
    for u, v in remaining_edges:
        remaining_adj[u].add(v)
        remaining_adj[v].add(u)
    
    best_left = None
    best_right = None
    best_size = 0
    
    # Strategy: for each vertex, try to form biclique with it on left
    # Pick right side as common neighbors
    vertices_with_edges = [v for v in range(n) if remaining_adj[v]]
    
    # Sort by degree descending for efficiency
    vertices_with_edges.sort(key=lambda v: -len(remaining_adj[v]))
    
    for u in vertices_with_edges[:15]:  # Try top 15 vertices
        # Right side candidates: neighbors of u in remaining graph
        right_candidates = list(remaining_adj[u])
        if not right_candidates:
            continue
        
        # Try to find a large common-neighbor set
        # For biclique [u] x R, all edges (u, r) for r in R must exist
        # But we can also have multiple left vertices
        
        # Better: try to find L and R such that all cross edges exist
        # Start with L = {u}, R = neighbors of u
        # Then try to add more left vertices that connect to all of R
        
        R = set(right_candidates)
        L = {u}
        
        # Try to expand L: add vertices that connect to all of R
        for v in vertices_with_edges:
            if v == u:
                continue
            if R.issubset(remaining_adj[v]):
                L.add(v)
        
        # Now try to expand R: add vertices that connect to all of L
        changed = True
        while changed:
            changed = False
            for v in vertices_with_edges:
                if v in L:
                    continue
                if L.issubset(remaining_adj[v]):
                    R.add(v)
                    changed = True
        
        size = len(L) * len(R)
        if size > best_size:
            best_size = size
            best_left = list(L)
            best_right = list(R)
    
    # Also try: pick two high-degree vertices, find common neighbors
    if len(vertices_with_edges) >= 2:
        for i in range(min(10, len(vertices_with_edges))):
            u = vertices_with_edges[i]
            for j in range(i+1, min(10, len(vertices_with_edges))):
                v = vertices_with_edges[j]
                common = remaining_adj[u] & remaining_adj[v]
                if len(common) >= 2:
                    L = {u, v}
                    R = set(common)
                    # Try to expand
                    changed = True
                    while changed:
                        changed = False
                        for w in vertices_with_edges:
                            if w in L:
                                continue
                            if R.issubset(remaining_adj[w]):
                                L.add(w)
                                changed = True
                    changed = True
                    while changed:
                        changed = False
                        for w in vertices_with_edges:
                            if w in R:
                                continue
                            if L.issubset(remaining_adj[w]):
                                R.add(w)
                                changed = True
                    size = len(L) * len(R)
                    if size > best_size:
                        best_size = size
                        best_left = list(L)
                        best_right = list(R)
    
    if best_left is None or best_right is None:
        return None, None
    
    # Verify all cross edges exist in remaining
    valid_left = []
    for u in best_left:
        valid_right = []
        for v in best_right:
            if u == v:
                continue
            e = (min(u,v), max(u,v))
            if e in edge_set:
                valid_right.append(v)
        if valid_right:
            valid_left.append((u, valid_right))
    
    if not valid_left:
        return None, None
    
    # Build the actual biclique from verified edges
    left_set = set(u for u, _ in valid_left)
    right_set = set(v for _, vs in valid_left for v in vs)
    
    # Verify it's a valid biclique
    final_left = []
    for u in left_set:
        u_right = []
        for v in right_set:
            if u == v:
                continue
            e = (min(u,v), max(u,v))
            if e in edge_set:
                u_right.append(v)
        if u_right:
            final_left.append(u)
    
    final_right = []
    for v in right_set:
        v_left = []
        for u in final_left:
            e = (min(u,v), max(u,v))
            if e in edge_set:
                v_left.append(u)
        if v_left:
            final_right.append(v)
    
    if not final_left or not final_right:
        return None, None
    
    return final_left, final_right

def _greedy_partition(edges, n, adj, deadline):
    """Greedy max-biclique extraction."""
    remaining = list(edges)
    remaining_set = set((min(u,v), max(u,v)) for u, v in remaining)
    bicliques = []
    
    while remaining and time.time() < deadline:
        left, right = _find_max_biclique(remaining, adj, n)
        
        if left is None or right is None:
            # Fallback: star partition for remaining
            for u in range(n):
                u_edges = [(u, v) for u2, v in remaining if u2 == u] + [(v, u) for u2, v in remaining if v == u]
                if u_edges:
                    neighbors = [v for _, v in u_edges]
                    bicliques.append([[int(u)], [int(v) for v in neighbors]])
            break
        
        # Remove edges covered by this biclique
        covered = set()
        for u in left:
            for v in right:
                e = (min(u,v), max(u,v))
                if e in remaining_set:
                    covered.add(e)
        
        if not covered:
            # Fallback
            for u in range(n):
                u_edges = [(u, v) for u2, v in remaining if u2 == u] + [(v, u) for u2, v in remaining if v == u]
                if u_edges:
                    neighbors = [v for _, v in u_edges]
                    bicliques.append([[int(u)], [int(v) for v in neighbors]])
            break
        
        remaining = [e for e in remaining if (min(e[0],e[1]), max(e[0],e[1])) not in covered]
        remaining_set -= covered
        bicliques.append([[int(u) for u in left], [int(v) for v in right]])
    
    return bicliques

def _star_partition(edges, n):
    """Fallback: per-vertex star partition."""
    bicliques = []
    for u in range(n):
        neighbors = [v for a, v in edges if a == u] + [a for a, v in edges if v == u]
        if neighbors:
            bicliques.append([[int(u)], [int(v) for v in neighbors]])
    return bicliques

def _validate(bicliques, edges, n):
    """Validate the biclique partition."""
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    covered = set()
    
    for left, right in bicliques:
        if not left or not right:
            return False
        # Check sides are disjoint
        if set(left) & set(right):
            return False
        # Check all cross edges exist
        for u in left:
            for v in right:
                e = (min(u,v), max(u,v))
                if e not in edge_set:
                    return False
                if e in covered:
                    return False  # Edge covered twice
                covered.add(e)
    
    # Check all edges covered exactly once
    if covered != edge_set:
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    edges = P['edges']
    n = P['n']
    adj = _build_adj(edges, n)
    
    # Try greedy partition with different random seeds for variety
    best_bicliques = None
    best_count = float('inf')
    
    # First try deterministic greedy
    bicliques = _greedy_partition(edges, n, adj, deadline)
    if bicliques and _validate(bicliques, edges, n):
        best_bicliques = bicliques
        best_count = len(bicliques)
    
    # Try multiple random restarts with shuffled edge orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Shuffle edges to get different greedy behavior
        shuffled = list(edges)
        random.shuffle(shuffled)
        
        bicliques = _greedy_partition(shuffled, n, adj, deadline)
        if bicliques and _validate(bicliques, edges, n):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_bicliques = bicliques
    
    # If greedy didn't work well, try star partition as fallback
    if best_bicliques is None or best_count > 20:
        star = _star_partition(edges, n)
        if _validate(star, edges, n):
            if best_bicliques is None or len(star) < best_count:
                best_bicliques = star
                best_count = len(star)
    
    if best_bicliques is None:
        # Last resort
        best_bicliques = _star_partition(edges, n)
    
    # Ensure correct types
    result = []
    for left, right in best_bicliques:
        result.append([list(int(x) for x in left), list(int(x) for x in right)])
    
    return result
# EVOLVE-BLOCK-END

