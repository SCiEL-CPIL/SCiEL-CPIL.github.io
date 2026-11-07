"""Model-and-optimize biclique partition for the pinned graph."""
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _validate(bicliques, edges_set):
    if not bicliques:
        return len(edges_set) == 0
    seen = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            if not isinstance(u, int) or isinstance(u, bool):
                return False
        for v in right:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                e = _edge_key(u, v)
                if e not in edges_set:
                    return False
                if e in seen:
                    return False
                seen.add(e)
    return seen == edges_set


def _star_partition(adj):
    result = []
    for u in range(len(adj)):
        neighbors = [v for v in adj[u]]
        if neighbors:
            result.append([neighbors, [u]])
    return result


def _find_maximal_biclique(adj, anchor, remaining_edges):
    """Find a maximal biclique containing the anchor vertex using only remaining edges."""
    rem_set = set(remaining_edges)
    
    # Start with anchor on left, all its neighbors in remaining edges on right
    left = {anchor}
    right = set()
    for u, v in rem_set:
        if u == anchor:
            right.add(v)
        elif v == anchor:
            right.add(u)
    
    if not right:
        return None, None
    
    # Iteratively refine to get maximal biclique
    for _ in range(50):
        changed = False
        
        # Filter right: must connect to all left
        new_right = set()
        for v in right:
            ok = all(_edge_key(u, v) in rem_set for u in left)
            if ok:
                new_right.add(v)
        if new_right != right:
            right = new_right
            changed = True
        
        # Filter left: must connect to all right
        new_left = set()
        for u in left:
            ok = all(_edge_key(u, v) in rem_set for v in right)
            if ok:
                new_left.add(u)
        if new_left != left:
            left = new_left
            changed = True
        
        if not changed:
            break
    
    if not left or not right:
        return None, None
    
    # Expand left
    for u in range(P['n']):
        if u not in left:
            ok = all(_edge_key(u, v) in rem_set for v in right)
            if ok:
                left.add(u)
    
    # Expand right
    for v in range(P['n']):
        if v not in right:
            ok = all(_edge_key(u, v) in rem_set for u in left)
            if ok:
                right.add(v)
    
    return left, right


def _greedy_partition(adj, edges_set, deadline, random_order=False, seed=None):
    """Greedy biclique partition with optional randomization."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    remaining = set(edges_set)
    bicliques = []
    
    while remaining and time.time() < deadline:
        # Compute degrees in remaining graph
        deg = {}
        for u, v in remaining:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        
        if not deg:
            break
        
        # Select anchor candidates
        if random_order:
            # Random selection from top-degree vertices
            sorted_anchors = sorted(deg.keys(), key=lambda x: -deg[x])
            k = min(10, len(sorted_anchors))
            candidates = rng.sample(sorted_anchors[:k], min(k, len(sorted_anchors[:k])))
        else:
            sorted_anchors = sorted(deg.keys(), key=lambda x: -deg[x])
            candidates = sorted_anchors[:15]
        
        best_left, best_right, best_size = None, None, 0
        
        for anchor in candidates:
            left, right = _find_maximal_biclique(adj, anchor, remaining)
            if left is None:
                continue
            size = len(left) * len(right)
            if size > best_size:
                best_size = size
                best_left = left
                best_right = right
        
        if best_left is None:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard(_edge_key(u, v))
            continue
        
        # Remove covered edges
        for u in best_left:
            for v in best_right:
                remaining.discard(_edge_key(u, v))
        
        bicliques.append([list(best_left), list(best_right)])
    
    # Leftover edges
    for e in remaining:
        bicliques.append([[e[0]], [e[1]]])
    
    return bicliques


def _try_merge_all(bicliques, edges_set, deadline):
    """Try to merge pairs of bicliques until no more merges possible."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bicliques)
        for i in range(n):
            if i >= n:
                break
            for j in range(i + 1, n):
                if j >= n:
                    break
                li, ri = bicliques[i][0], bicliques[i][1]
                lj, rj = bicliques[j][0], bicliques[j][1]
                
                new_left = set(li) | set(lj)
                new_right = set(ri) | set(rj)
                
                # Check all cross pairs are edges
                valid = True
                for u in new_left:
                    for v in new_right:
                        if u == v or _edge_key(u, v) not in edges_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check coverage matches
                covered_i = set(_edge_key(u, v) for u in li for v in ri)
                covered_j = set(_edge_key(u, v) for u in lj for v in rj)
                covered_new = set(_edge_key(u, v) for u in new_left for v in new_right)
                
                if covered_new == covered_i | covered_j:
                    bicliques[i] = [list(new_left), list(new_right)]
                    del bicliques[j]
                    improved = True
                    break
            if improved:
                break
    
    return bicliques


def _local_search(bicliques, edges_set, deadline):
    """Local search: try vertex moves between bicliques."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bicliques)
        
        # Try moving a vertex from one biclique's side to another's side
        for i in range(n):
            li, ri = bicliques[i][0], bicliques[i][1]
            for j in range(n):
                if i == j:
                    continue
                lj, rj = bicliques[j][0], bicliques[j][1]
                
                # Try moving a vertex from lj to ri
                for u in list(lj):
                    if u in ri:
                        continue
                    ok = all(_edge_key(u, v) in edges_set for v in li)
                    if ok:
                        new_ri = ri + [u]
                        new_lj = [x for x in lj if x != u]
                        if new_lj:
                            ok2 = all(_edge_key(a, b) in edges_set for a in new_lj for b in rj)
                            if ok2:
                                cov_i = set(_edge_key(a, b) for a in li for b in new_ri)
                                cov_j = set(_edge_key(a, b) for a in new_lj for b in rj)
                                if not (cov_i & cov_j):
                                    bicliques[i] = [li, new_ri]
                                    bicliques[j] = [new_lj, rj]
                                    improved = True
                                    break
                if improved:
                    break
            
            # Try moving a vertex from rj to li
            for u in list(rj):
                if u in li:
                    continue
                ok = all(_edge_key(u, v) in edges_set for v in ri)
                if ok:
                    new_li = li + [u]
                    new_rj = [x for x in rj if x != u]
                    if new_rj:
                        ok2 = all(_edge_key(a, b) in edges_set for a in new_li for b in ri)
                        if ok2:
                            cov_i = set(_edge_key(a, b) for a in new_li for b in ri)
                            cov_j = set(_edge_key(a, b) for a in lj for b in new_rj)
                            if not (cov_i & cov_j):
                                bicliques[i] = [new_li, ri]
                                bicliques[j] = [lj, new_rj]
                                improved = True
                                break
            if improved:
                break
        
        # Try merging after moves
        if not improved:
            bicliques = _try_merge_all(bicliques, edges_set, deadline)
            if len(bicliques) < n:
                improved = True
    
    return bicliques


def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    adj = [set() for _ in range(P['n'])]
    edges_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edges_set.add(_edge_key(u, v))
    
    # Fallback
    fallback = _star_partition(adj)
    best = fallback
    best_count = len(fallback)
    
    # Multiple restarts with different strategies
    for attempt in range(8):
        if time.time() > deadline - 5:
            break
        
        if attempt == 0:
            # Deterministic greedy
            bicliques = _greedy_partition(adj, edges_set, deadline - 5, random_order=False)
        elif attempt == 1:
            # Random greedy
            bicliques = _greedy_partition(adj, edges_set, deadline - 5, random_order=True, seed=42 + attempt)
        else:
            # More random restarts
            bicliques = _greedy_partition(adj, edges_set, deadline - 5, random_order=True, seed=100 + attempt)
        
        # Local search
        if time.time() < deadline - 10:
            bicliques = _local_search(bicliques, edges_set, deadline - 10)
        
        if _validate(bicliques, edges_set) and len(bicliques) < best_count:
            best = bicliques
            best_count = len(bicliques)
    
    # Final validation
    if not _validate(best, edges_set):
        best = fallback
    
    # Ensure correct types
    result = []
    for left, right in best:
        result.append([[int(u) for u in left], [int(v) for v in right]])
    
    return result
