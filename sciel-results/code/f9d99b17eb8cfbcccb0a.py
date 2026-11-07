"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _edge_key(u, v):
    return (min(u, v), max(u, v))


def _is_biclique(left, right, adj):
    """Check if (left, right) forms a valid biclique."""
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
    return True


def _find_biclique_from_seed(seed_left, seed_right, adj, remaining_edges):
    """Expand a seed biclique as much as possible using only remaining edges."""
    left_set = set(seed_left)
    right_set = set(seed_right)
    
    # Verify seed is valid with remaining edges
    for u in left_set:
        for v in right_set:
            if _edge_key(u, v) not in remaining_edges:
                return None
    
    # Iteratively expand
    changed = True
    while changed:
        changed = False
        # Try to add to left
        for v in range(P['n']):
            if v in left_set or v in right_set:
                continue
            can_add = True
            for r in right_set:
                if _edge_key(v, r) not in remaining_edges:
                    can_add = False
                    break
            if can_add:
                left_set.add(v)
                changed = True
        # Try to add to right
        for v in range(P['n']):
            if v in left_set or v in right_set:
                continue
            can_add = True
            for l in left_set:
                if _edge_key(v, l) not in remaining_edges:
                    can_add = False
                    break
            if can_add:
                right_set.add(v)
                changed = True
    
    if not left_set or not right_set:
        return None
    return (list(left_set), list(right_set))


def _find_best_biclique(adj, remaining_edges, deadline, max_anchors=30):
    """Find the largest biclique among remaining edges using multiple seeds."""
    best_biclique = None
    best_size = 0
    
    # Compute remaining degrees
    deg_remaining = {}
    for (u, v) in remaining_edges:
        deg_remaining[u] = deg_remaining.get(u, 0) + 1
        deg_remaining[v] = deg_remaining.get(v, 0) + 1
    
    if not deg_remaining:
        return None
    
    # Sort vertices by remaining degree
    sorted_vertices = sorted(deg_remaining.keys(), key=lambda x: -deg_remaining[x])
    
    # Try multiple seed strategies
    seeds_tried = set()
    
    for anchor in sorted_vertices[:max_anchors]:
        if time.time() > deadline:
            break
        
        # Strategy 1: seed with just the anchor and its top neighbor
        neighbors = [v for v in adj[anchor] if _edge_key(anchor, v) in remaining_edges]
        if not neighbors:
            continue
        
        # Sort neighbors by their remaining degree
        neighbors.sort(key=lambda v: -deg_remaining.get(v, 0))
        
        for nb in neighbors[:5]:
            seed_key = (anchor, nb)
            if seed_key in seeds_tried:
                continue
            seeds_tried.add(seed_key)
            
            bl = _find_biclique_from_seed([anchor], [nb], adj, remaining_edges)
            if bl:
                size = len(bl[0]) * len(bl[1])
                if size > best_size:
                    best_size = size
                    best_biclique = bl
    
    return best_biclique


def _greedy_partition(adj, deadline):
    """Greedy: repeatedly find a large biclique among remaining edges."""
    remaining = set()
    for a, b in P['edges']:
        remaining.add(_edge_key(a, b))
    
    bicliques = []
    while remaining:
        if time.time() > deadline:
            break
        
        bl = _find_best_biclique(adj, remaining, deadline)
        
        if bl is None:
            # Fallback: take one edge
            e = next(iter(remaining))
            bl = ([e[0]], [e[1]])
        
        l, r = bl
        bicliques.append((l, r))
        
        # Remove all cross edges from remaining
        for u in l:
            for v in r:
                remaining.discard(_edge_key(u, v))
    
    return bicliques


def _merge_pass(bicliques, adj, deadline):
    """Try to merge pairs of bicliques if their union forms a valid biclique."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bicliques)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                l1, r1 = bicliques[i]
                l2, r2 = bicliques[j]
                
                # Try all 4 merge orientations
                merged = None
                for (new_l, new_r) in [
                    (set(l1) | set(l2), set(r1) & set(r2)),
                    (set(l1) & set(l2), set(r1) | set(r2)),
                    (set(l1) | set(r2), set(r1) & set(l2)),
                    (set(l2) | set(r1), set(r2) & set(l1)),
                ]:
                    if not new_l or not new_r:
                        continue
                    if new_l & new_r:
                        continue
                    # Verify all cross pairs are edges
                    if _is_biclique(new_l, new_r, adj):
                        merged = (list(new_l), list(new_r))
                        break
                
                if merged:
                    # Check that merged covers exactly the union of edges from i and j
                    edges_covered = set()
                    for u in merged[0]:
                        for v in merged[1]:
                            edges_covered.add(_edge_key(u, v))
                    old_edges = set()
                    for u in l1:
                        for v in r1:
                            old_edges.add(_edge_key(u, v))
                    for u in l2:
                        for v in r2:
                            old_edges.add(_edge_key(u, v))
                    if edges_covered == old_edges:
                        bicliques[i] = merged
                        bicliques.pop(j)
                        changed = True
                        break
            if changed:
                break
    return bicliques


def _absorb_pass(bicliques, adj, deadline):
    """Try to absorb edges from one biclique into another if valid."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(bicliques)
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(n):
                if i == j:
                    continue
                l1, r1 = bicliques[i]
                l2, r2 = bicliques[j]
                
                # Try to expand biclique i by adding vertices from biclique j
                # Add a vertex from l2 to r1 if it connects to all of l1
                for v in l2:
                    if v in l1 or v in r1:
                        continue
                    if all(_edge_key(v, u) in set(_edge_key(a, b) for a, b in P['edges']) for u in l1):
                        # Check if v can be added to r1
                        can_add = True
                        for u in l1:
                            if v not in adj[u]:
                                can_add = False
                                break
                        if can_add:
                            new_r1 = set(r1) | {v}
                            new_l1 = set(l1)
                            if not (new_l1 & new_r1) and _is_biclique(new_l1, new_r1, adj):
                                # Verify the new biclique only covers edges that were in biclique i or j
                                new_edges = set()
                                for u in new_l1:
                                    for v2 in new_r1:
                                        new_edges.add(_edge_key(u, v2))
                                old_i_edges = set()
                                for u in l1:
                                    for v2 in r1:
                                        old_i_edges.add(_edge_key(u, v2))
                                old_j_edges = set()
                                for u in l2:
                                    for v2 in r2:
                                        old_j_edges.add(_edge_key(u, v2))
                                if new_edges <= (old_i_edges | old_j_edges):
                                    bicliques[i] = (list(new_l1), list(new_r1))
                                    # Remove absorbed edges from biclique j
                                    new_l2 = set(l2)
                                    new_r2 = set(r2)
                                    if v in l2:
                                        new_l2.discard(v)
                                    # Check if biclique j is still valid
                                    if new_l2 and new_r2 and not (new_l2 & new_r2) and _is_biclique(new_l2, new_r2, adj):
                                        bicliques[j] = (list(new_l2), list(new_r2))
                                    else:
                                        bicliques.pop(j)
                                    changed = True
                                    break
                if changed:
                    break
            if changed:
                break
    return bicliques


def _refine_pass(bicliques, adj, deadline):
    """Try to split bicliques and recombine for better coverage."""
    # This is a placeholder for more sophisticated local search
    return bicliques


def _validate(bicliques, adj):
    """Validate that the partition is correct."""
    all_edges = set()
    for a, b in P['edges']:
        all_edges.add(_edge_key(a, b))
    
    covered = set()
    for l, r in bicliques:
        if not l or not r:
            return False
        if set(l) & set(r):
            return False
        for u in l:
            for v in r:
                if v not in adj[u]:
                    return False
                covered.add(_edge_key(u, v))
    
    if covered != all_edges:
        return False
    
    return True


def _to_result(bicliques):
    """Convert to the required format with proper types."""
    result = []
    for l, r in bicliques:
        result.append([list(l), list(r)])
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    adj = _build_adj()
    
    # Run greedy with multiple random restarts
    best_bicliques = None
    best_count = float('inf')
    
    for restart in range(100):
        if time.time() > deadline - 3:
            break
        random.seed(restart)
        bl = _greedy_partition(adj, deadline - 3)
        if bl:
            bl = _merge_pass(bl, adj, deadline - 2)
            bl = _absorb_pass(bl, adj, deadline - 1)
            if _validate(bl, adj) and len(bl) < best_count:
                best_count = len(bl)
                best_bicliques = bl
    
    if best_bicliques is None:
        # Fallback: star partition (always valid)
        best_bicliques = []
        for u in range(P['n']):
            neighbors = [v for a, b in P['edges'] if (a == u and b not in [x for x in [a2 for a2, b2 in P['edges'] if a2 == u]])]
            # Simpler: get all neighbors
            neighbors = list(adj[u])
            if neighbors:
                best_bicliques.append(([u], neighbors))
    
    return _to_result(best_bicliques)
# EVOLVE-BLOCK-END
