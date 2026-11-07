"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_max_biclique(adj, remaining_edges):
    """Find a large biclique in the subgraph defined by remaining_edges."""
    if not remaining_edges:
        return None, None
    
    # Count how many remaining edges each vertex is incident to
    deg = {}
    for u, v in remaining_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    
    if not deg:
        return None, None
    
    # Try multiple anchor vertices and pick the best biclique
    best_left, best_right, best_size = None, None, 0
    
    # Sort by degree descending, try top candidates
    sorted_vertices = sorted(deg.keys(), key=lambda x: -deg[x])
    max_anchors = min(len(sorted_vertices), 20)
    
    for anchor in sorted_vertices[:max_anchors]:
        # Its neighbors in the remaining edge set
        candidates = set()
        for u, v in remaining_edges:
            if u == anchor:
                candidates.add(v)
            elif v == anchor:
                candidates.add(u)
        
        if not candidates:
            continue
        
        # Greedily build biclique: start with anchor on left, try to add vertices
        left = {anchor}
        right = set(candidates)
        
        # Iteratively refine: keep only right vertices connected to all left, and vice versa
        changed = True
        while changed:
            changed = False
            # Filter right: must be connected to all in left
            new_right = set()
            for v in right:
                if all(v in adj[u] for u in left):
                    new_right.add(v)
            if new_right != right:
                right = new_right
                changed = True
            
            # Filter left: must be connected to all in right
            new_left = set()
            for u in left:
                if all(u in adj[v] for v in right):
                    new_left.add(u)
            if new_left != left:
                left = new_left
                changed = True
        
        if not left or not right:
            continue
        
        # Try to expand left by adding vertices connected to all of right
        for u in range(P['n']):
            if u not in left and u in deg:
                if all(u in adj[v] for v in right):
                    left.add(u)
        
        # Try to expand right by adding vertices connected to all of left
        for v in range(P['n']):
            if v not in right and v in deg:
                if all(v in adj[u] for u in left):
                    right.add(v)
        
        # Calculate size of this biclique (number of edges it covers)
        size = len(left) * len(right)
        if size > best_size:
            best_size = size
            best_left = list(left)
            best_right = list(right)
    
    return best_left, best_right

def _validate_bicliques(bicliques, edges_set):
    """Validate that bicliques form a valid partition."""
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                edge = tuple(sorted((u, v)))
                if edge not in edges_set:
                    return False
                if edge in covered:
                    return False
                covered.add(edge)
    
    return covered == edges_set

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    adj = _build_adjacency()
    edges_set = set(tuple(sorted(e)) for e in P['edges'])
    
    # Fallback: star decomposition
    def star_partition():
        result = []
        for u in range(P['n']):
            neighbors = [v for v in adj[u]]
            if neighbors:
                result.append([list(neighbors), [u]])
        return result
    
    fallback = star_partition()
    best = fallback
    
    # Greedy biclique partition
    remaining = set(edges_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        left, right = _find_max_biclique(adj, remaining)
        
        if left is None or right is None:
            # Fallback for this edge
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard((u, v))
            continue
        
        # Verify this biclique only uses remaining edges
        valid = True
        for u in left:
            for v in right:
                edge = tuple(sorted((u, v)))
                if edge not in remaining:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Shrink to only remaining edges
            left = [u for u in left if all(tuple(sorted((u, v))) in remaining for v in right)]
            right = [v for v in right if all(tuple(sorted((u, v))) in remaining for u in left)]
        
        if not left or not right:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard((u, v))
            continue
        
        # Remove covered edges
        for u in left:
            for v in right:
                edge = tuple(sorted((u, v)))
                remaining.discard(edge)
        
        bicliques.append([left, right])
    
    # Handle leftover edges with individual bicliques
    for edge in remaining:
        u, v = edge
        bicliques.append([[u], [v]])
    
    # Local search: try to merge small bicliques or reassign edges
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try to merge two bicliques if their union is also a valid biclique
            n_bicliques = len(bicliques)
            for i in range(n_bicliques):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, n_bicliques):
                    if time.time() > deadline - 10:
                        break
                    li, ri = bicliques[i][0], bicliques[i][1]
                    lj, rj = bicliques[j][0], bicliques[j][1]
                    
                    # Try merging: new_left = li ∪ lj, new_right = ri ∪ rj
                    # Check if all cross pairs are edges
                    new_left = set(li) | set(lj)
                    new_right = set(ri) | set(rj)
                    
                    # Check validity
                    valid_merge = True
                    for u in new_left:
                        for v in new_right:
                            if u == v:
                                valid_merge = False
                                break
                            edge = tuple(sorted((u, v)))
                            if edge not in edges_set:
                                valid_merge = False
                                break
                        if not valid_merge:
                            break
                    if not valid_merge:
                        continue
                    
                    # Check that merged biclique covers exactly the union of edges from both
                    covered_i = set()
                    for u in li:
                        for v in ri:
                            covered_i.add(tuple(sorted((u, v))))
                    covered_j = set()
                    for u in lj:
                        for v in rj:
                            covered_j.add(tuple(sorted((u, v))))
                    
                    covered_new = set()
                    for u in new_left:
                        for v in new_right:
                            covered_new.add(tuple(sorted((u, v))))
                    
                    if covered_new == covered_i | covered_j:
                        # Merge is valid
                        bicliques[i] = [list(new_left), list(new_right)]
                        del bicliques[j]
                        improved = True
                        break
                if improved:
                    break
    
    # Validate
    if _validate_bicliques(bicliques, edges_set):
        best = bicliques
    else:
        best = fallback
    
    return best

# EVOLVE-BLOCK-END
