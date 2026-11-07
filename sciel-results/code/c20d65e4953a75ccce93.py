"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_valid_biclique(left, right, adj):
    if not left or not right:
        return False
    if set(left) & set(right):
        return False
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
    return True

def _edge_set(biclique):
    left, right = biclique
    edges = set()
    for u in left:
        for v in right:
            edges.add(frozenset((u, v)))
    return edges

def _partition_valid(bicliques, adj, all_edges):
    if not bicliques:
        return False
    covered = set()
    for b in bicliques:
        left, right = b
        if not left or not right:
            return False
        if set(left) & set(right):
            return False
        for u in left:
            for v in right:
                e = frozenset((u, v))
                if e not in all_edges:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == all_edges

def _greedy_max_biclique(adj, remaining_edges, n):
    """Find a large biclique from remaining edges using greedy heuristic."""
    if not remaining_edges:
        return None
    
    # Build adjacency within remaining edges
    # For each vertex, track which neighbors are still connected via remaining edges
    # Use a simple greedy: pick a vertex, extend to form biclique
    
    best_biclique = None
    best_size = 0
    
    # Get all vertices that appear in remaining edges
    involved = set()
    for e in remaining_edges:
        involved.update(e)
    
    for start in involved:
        # Try to build a biclique starting with start on left
        left = {start}
        # Right candidates: neighbors of start that have remaining edges
        right = set()
        for e in remaining_edges:
            if start in e:
                other = (e - {start}).pop()
                right.add(other)
        
        if not right:
            continue
        
        # Greedily expand: add vertices to left/right if they connect to all on other side
        changed = True
        while changed:
            changed = False
            # Try adding to left
            for v in list(right):
                # Check if v can be moved to left (connects to all in left)
                # Actually, let's try a different approach: 
                # Find common neighbors
                pass
        
        # Simpler: just use the star from start
        size = len(left) * len(right)
        if size > best_size:
            best_size = size
            best_biclique = [list(left), list(right)]
    
    # Also try: for each edge in remaining, try to expand it
    for e in list(remaining_edges)[:50]:  # Limit for speed
        u, v = tuple(e)
        left = {u}
        right = {v}
        changed = True
        while changed:
            changed = False
            # Add to left: vertices connected to all in right
            for w in involved:
                if w not in left and w not in right:
                    if all(x in adj[w] for x in right):
                        left.add(w)
                        changed = True
            # Add to right: vertices connected to all in left
            for w in involved:
                if w not in left and w not in right:
                    if all(x in adj[w] for x in left):
                        right.add(w)
                        changed = True
        
        size = len(left) * len(right)
        if size > best_size:
            best_size = size
            best_biclique = [list(left), list(right)]
    
    return best_biclique

def _greedy_partition(adj, all_edges, deadline):
    """Greedy: repeatedly find max biclique, remove its edges."""
    remaining = set(all_edges)
    bicliques = []
    
    while remaining and time.time() < deadline:
        b = _greedy_max_biclique(adj, remaining, P['n'])
        if b is None:
            # Fallback: single edge
            e = next(iter(remaining))
            u, v = tuple(e)
            b = [[u], [v]]
        
        left, right = b
        # Verify it's valid
        if not _is_valid_biclique(left, right, adj):
            # Fallback: single edge
            e = next(iter(remaining))
            u, v = tuple(e)
            b = [[u], [v]]
            left, right = b
        
        bicliques.append([left, right])
        
        # Remove covered edges
        for u in left:
            for v in right:
                remaining.discard(frozenset((u, v)))
    
    return bicliques

def _local_search(bicliques, adj, all_edges, deadline):
    """Try to merge pairs of bicliques to reduce count."""
    bicliques = [b[:] for b in bicliques]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bicliques)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                bi, bj = bicliques[i], bicliques[j]
                li, ri = bi
                lj, rj = bj
                
                # Try all 4 combinations of merging sides
                candidates = []
                # Merge: left_i ∪ left_j, right_i ∪ right_j
                candidates.append(([li, ri, lj, rj], (li + lj, ri + rj)))
                # left_i ∪ right_j, right_i ∪ left_j
                candidates.append(([li, rj, ri, lj], (li + rj, ri + lj)))
                # left_i, right_i ∪ right_j  (if left_i compatible)
                # Actually, merging means taking union of vertices on each side
                # Option 1: (L1∪L2, R1∪R2)
                # Option 2: (L1∪R2, R1∪L2)
                
                for _, (new_left, new_right) in candidates:
                    # Deduplicate
                    new_left = list(set(new_left))
                    new_right = list(set(new_right))
                    if not new_left or not new_right:
                        continue
                    if set(new_left) & set(new_right):
                        continue
                    if _is_valid_biclique(new_left, new_right, adj):
                        # Check that the merged biclique covers exactly the union of edges
                        # (it should, since each original biclique's edges are a subset)
                        merged = [new_left, new_right]
                        new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
                        new_bicliques.append(merged)
                        if _partition_valid(new_bicliques, adj, all_edges):
                            bicliques = new_bicliques
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    adj = _build_adj()
    all_edges = set(frozenset(e) for e in P['edges'])
    
    # Phase 1: Greedy partition
    bicliques = _greedy_partition(adj, all_edges, deadline)
    
    # Phase 2: Local search (merge)
    if time.time() < deadline:
        bicliques = _local_search(bicliques, adj, all_edges, deadline)
    
    # Validate final result
    if not _partition_valid(bicliques, adj, all_edges):
        # Fallback: seed solution
        bicliques = [[[u], [v for a, v in P['edges'] if a == u]] 
                     for u in range(P['n']) 
                     if any(a == u for a, v in P['edges'])]
    
    # Ensure proper types
    result = []
    for b in bicliques:
        left = [int(v) for v in b[0]]
        right = [int(v) for v in b[1]]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
