"""Biclique partition via greedy set-cover on maximal bicliques."""
import time
import itertools

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_maximal_bicliques(adj, n, deadline):
    """Find a good set of maximal bicliques using a greedy approach."""
    bicliques = []
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    
    # For each vertex, try to build a biclique centered on it
    for center in range(n):
        if time.time() > deadline:
            break
        # Try to find a large biclique with center on left
        # Start with all neighbors of center
        left = {center}
        # Common neighbors of all left vertices
        common = set(adj[center])
        # Try to expand: for each potential right vertex, check if it connects to all left
        # Greedy: add right vertices that have the most connections to left
        right_candidates = sorted(common, key=lambda v: -len(adj[v] & left))
        right = set()
        for v in right_candidates:
            if v in left:
                continue
            # v must connect to all vertices in left
            if all(v in adj[u] for u in left):
                right.add(v)
        if left and right:
            bicliques.append((frozenset(left), frozenset(right)))
        
        # Also try center on right
        right2 = {center}
        common2 = set(adj[center])
        left_candidates = sorted(common2, key=lambda v: -len(adj[v] & right2))
        left2 = set()
        for v in left_candidates:
            if v in right2:
                continue
            if all(v in adj[u] for u in right2):
                left2.add(v)
        if left2 and right2:
            bicliques.append((frozenset(left2), frozenset(right2)))
    
    # Also try pairs of vertices as seeds for bicliques
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i+1, n):
            if time.time() > deadline:
                break
            if (i, j) not in edge_set and (j, i) not in edge_set:
                continue
            # i and j are adjacent, try to form biclique with i on left, j on right
            # or both on same side
            # Try: left={i}, right={j}, expand
            left = {i}
            right = {j}
            # Expand left: add vertices connected to all in right
            changed = True
            while changed:
                changed = False
                for v in range(n):
                    if v not in left and v not in right:
                        if all(v in adj[u] for u in right):
                            left.add(v)
                            changed = True
            # Expand right: add vertices connected to all in left
            changed = True
            while changed:
                changed = False
                for v in range(n):
                    if v not in left and v not in right:
                        if all(v in adj[u] for u in left):
                            right.add(v)
                            changed = True
            if left and right:
                bicliques.append((frozenset(left), frozenset(right)))
    
    return bicliques

def _greedy_partition(adj, edge_set, bicliques, deadline):
    """Greedy set-cover: repeatedly pick the biclique covering most uncovered edges."""
    remaining = set(edge_set)
    partition = []
    
    while remaining and time.time() < deadline:
        best_biclique = None
        best_count = 0
        best_edges = set()
        
        for bl in bicliques:
            left, right = bl
            # Count how many of this biclique's cross edges are in remaining
            count = 0
            bl_edges = []
            for u in left:
                for v in right:
                    e = (min(u,v), max(u,v))
                    if e in remaining:
                        count += 1
                        bl_edges.append(e)
            if count > best_count:
                best_count = count
                best_biclique = bl
                best_edges = set(bl_edges)
            elif count == best_count and count > 0:
                # Tie-break: prefer larger bicliques
                if len(bl[0]) * len(bl[1]) > len(best_biclique[0]) * len(best_biclique[1]):
                    best_biclique = bl
                    best_edges = set(bl_edges)
        
        if best_count == 0:
            break
        
        partition.append(best_biclique)
        remaining -= best_edges
    
    # For any remaining edges, add them as individual bicliques
    for e in remaining:
        partition.append((frozenset({e[0]}), frozenset({e[1]})))
    
    return partition

def construct():
    deadline = time.time() + 80  # 80s budget, 10s safety margin
    
    adj = _build_adj()
    n = P['n']
    
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    
    # Find candidate bicliques
    bicliques = _find_maximal_bicliques(adj, n, deadline)
    
    # Greedy partition
    partition = _greedy_partition(adj, edge_set, bicliques, deadline)
    
    # Convert to required format: list of [left_list, right_list]
    result = []
    for left, right in partition:
        left_list = [int(v) for v in sorted(left)]
        right_list = [int(v) for v in sorted(right)]
        if left_list and right_list:
            result.append([left_list, right_list])
    
    # Validate: ensure no edge is covered twice
    # (The greedy should handle this, but double-check)
    covered = set()
    valid_result = []
    for bic in result:
        left, right = bic[0], bic[1]
        ok = True
        for u in left:
            for v in right:
                e = (min(u,v), max(u,v))
                if e in covered:
                    ok = False
                    break
                if e not in edge_set:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            for u in left:
                for v in right:
                    covered.add((min(u,v), max(u,v)))
            valid_result.append(bic)
    
    # Add any missing edges as single-edge bicliques
    for e in edge_set:
        if e not in covered:
            valid_result.append([[int(e[0])], [int(e[1])]])
            covered.add(e)
    
    return valid_result
# EVOLVE-BLOCK-END
