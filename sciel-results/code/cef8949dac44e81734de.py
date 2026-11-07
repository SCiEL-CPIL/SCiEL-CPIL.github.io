"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

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
    
    # Pick vertex with highest degree as anchor
    anchor = max(deg, key=lambda x: deg[x])
    
    # Its neighbors in the remaining edge set
    candidates = set()
    for u, v in remaining_edges:
        if u == anchor:
            candidates.add(v)
        elif v == anchor:
            candidates.add(u)
    
    if not candidates:
        return None, None
    
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
    
    return list(left), list(right)

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

def _local_search(bicliques, adj, edges_set, deadline):
    """Try to reduce number of bicliques by merging or splitting."""
    current = [list(bc) for bc in bicliques]
    best_count = len(current)
    
    while time.time() < deadline:
        improved = False
        
        # Try merging pairs of bicliques
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                if len(current) <= best_count:
                    break
                left1, right1 = current[i]
                left2, right2 = current[j]
                
                # Try all four combinations of merging
                for merge_type in range(4):
                    if merge_type == 0:
                        new_left = list(set(left1) | set(left2))
                        new_right = list(set(right1) & set(right2))
                    elif merge_type == 1:
                        new_left = list(set(left1) & set(left2))
                        new_right = list(set(right1) | set(right2))
                    elif merge_type == 2:
                        new_left = list(set(left1) | set(left2))
                        new_right = list(set(right1) | set(right2))
                    else:
                        new_left = list(set(left1) & set(left2))
                        new_right = list(set(right1) & set(right2))
                    
                    if not new_left or not new_right:
                        continue
                    
                    # Check if merged biclique is valid and covers only edges from the two original bicliques
                    valid = True
                    covered_edges = set()
                    for u in new_left:
                        for v in new_right:
                            if u == v:
                                valid = False
                                break
                            edge = tuple(sorted((u, v)))
                            if edge not in edges_set:
                                valid = False
                                break
                            if edge in covered_edges:
                                valid = False
                                break
                            covered_edges.add(edge)
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Check if covered edges are exactly the union of edges from biclique i and j
                    edges_i = set()
                    for u in left1:
                        for v in right1:
                            edges_i.add(tuple(sorted((u, v))))
                    edges_j = set()
                    for u in left2:
                        for v in right2:
                            edges_j.add(tuple(sorted((u, v))))
                    
                    if covered_edges != (edges_i | edges_j):
                        continue
                    
                    # Try the merge
                    new_bicliques = [bc for k, bc in enumerate(current) if k != i and k != j]
                    new_bicliques.append([new_left, new_right])
                    
                    if _validate_bicliques(new_bicliques, edges_set):
                        current = new_bicliques
                        best_count = len(current)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
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
    
    # Validate
    if _validate_bicliques(bicliques, edges_set):
        best = bicliques
    else:
        best = fallback
    
    # Local search to try to improve
    if time.time() < deadline:
        improved = _local_search(best, adj, edges_set, deadline)
        if _validate_bicliques(improved, edges_set) and len(improved) < len(best):
            best = improved
    
    # Ensure proper types
    result = []
    for left, right in best:
        result.append([[int(v) for v in left], [int(v) for v in right]])
    
    return result

# EVOLVE-BLOCK-END
