"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
sys.setrecursionlimit(10000)

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _biclique_edges_set(left, right):
    """Return the set of edges covered by a biclique."""
    edges = set()
    for u in left:
        for v in right:
            e = (min(u, v), max(u, v))
            edges.add(e)
    return edges

def _is_valid_biclique(left, right, adj):
    """Check that all cross pairs are edges."""
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
    return True

def _expand_biclique(left, right, adj, all_edges_set):
    """Expand a biclique to be maximal with respect to remaining edges."""
    left = set(left)
    right = set(right)
    changed = True
    while changed:
        changed = False
        # Expand left
        new_left = set()
        for u in range(P['n']):
            if u in left or u in right:
                continue
            if right.issubset(adj[u]):
                # Check u has edges to all in right that are in all_edges_set
                has_edge = True
                for v in right:
                    e = (min(u, v), max(u, v))
                    if e not in all_edges_set:
                        has_edge = False
                        break
                if has_edge:
                    new_left.add(u)
                    changed = True
        left |= new_left
        
        # Expand right
        new_right = set()
        for v in range(P['n']):
            if v in left or v in right:
                continue
            if left.issubset(adj[v]):
                has_edge = True
                for u in left:
                    e = (min(u, v), max(u, v))
                    if e not in all_edges_set:
                        has_edge = False
                        break
                if has_edge:
                    new_right.add(v)
                    changed = True
        right |= new_right
    
    right -= left
    left -= right
    return left, right

def _generate_candidate_bicliques(adj, all_edges_set, deadline, num_seeds=500):
    """Generate diverse candidate bicliques using random seeds."""
    candidates = []
    seen_edge_sets = set()
    n = P['n']
    
    # Seed from each edge
    edges_list = list(all_edges_set)
    random.shuffle(edges_list)
    
    for i in range(min(num_seeds, len(edges_list))):
        if time.time() > deadline:
            break
        e = edges_list[i]
        u0, v0 = e
        left, right = _expand_biclique({u0}, {v0}, adj, all_edges_set)
        if left and right:
            edge_set = _biclique_edges_set(left, right)
            if edge_set not in seen_edge_sets:
                seen_edge_sets.add(edge_set)
                candidates.append((left, right, edge_set))
    
    # Also try some random vertex pairs
    for _ in range(200):
        if time.time() > deadline:
            break
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if v in adj[u]:
            left, right = _expand_biclique({u}, {v}, adj, all_edges_set)
            if left and right:
                edge_set = _biclique_edges_set(left, right)
                if edge_set not in seen_edge_sets:
                    seen_edge_sets.add(edge_set)
                    candidates.append((left, right, edge_set))
    
    return candidates

def _algorithm_x(candidates, all_edges_set, deadline):
    """
    Solve minimum set cover using Algorithm X with Dancing Links.
    candidates: list of (left, right, edge_set)
    Returns: list of selected bicliques (left, right) or None
    """
    n = P['n']
    total_edges = len(all_edges_set)
    
    # Map each edge to an index
    edge_to_idx = {}
    for idx, e in enumerate(all_edges_set):
        edge_to_idx[e] = idx
    
    # Build incidence matrix: rows = candidates, columns = edges
    # Use a simple representation for Algorithm X
    # Each row: list of column indices it covers
    
    rows = []
    for left, right, edge_set in candidates:
        cols = []
        for e in edge_set:
            if e in edge_to_idx:
                cols.append(edge_to_idx[e])
        if cols:
            rows.append(cols)
    
    if not rows:
        return None
    
    # Simple greedy set cover as fallback if Algorithm X is too slow
    # But first try a bounded backtracking with Algorithm X style
    
    # For efficiency, use a simpler approach: greedy with backtracking limit
    # Since the graph is small (30 nodes, ~200 edges), we can use a more direct approach
    
    # Greedy set cover
    remaining = set(all_edges_set)
    selected = []
    available = list(candidates)
    
    while remaining:
        if time.time() > deadline:
            break
        # Find candidate that covers the most remaining edges
        best_idx = -1
        best_count = 0
        for i, (left, right, edge_set) in enumerate(available):
            count = len(edge_set & remaining)
            if count > best_count:
                best_count = count
                best_idx = i
            if count == 0:
                available[i] = None  # Mark as unusable
        
        if best_idx == -1 or best_count == 0:
            # Fallback: use single edge
            e = next(iter(remaining))
            u, v = e
            selected.append(({u}, {v}))
            remaining.discard(e)
        else:
            left, right, edge_set = available[best_idx]
            selected.append((left, right))
            remaining -= edge_set
            available[best_idx] = None
    
    if not remaining:
        return selected
    return None

def _validate(bicliques, adj, all_edges_set):
    """Check that bicliques form a valid partition."""
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    if covered != all_edges_set:
        return False
    return True

def _local_search(bicliques, adj, all_edges_set, deadline):
    """Try to merge bicliques to reduce count."""
    bicliques = [list(bc) for bc in bicliques]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(bicliques)):
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                
                # Try merge: left = li ∪ lj, right = ri ∩ rj
                if ri & rj:
                    new_left = set(li) | set(lj)
                    new_right = set(ri) & set(rj)
                    if new_left and new_right and _is_valid_biclique(new_left, new_right, adj):
                        merge_edges = _biclique_edges_set(new_left, new_right)
                        old_edges = _biclique_edges_set(li, ri) | _biclique_edges_set(lj, rj)
                        if merge_edges == old_edges:
                            bicliques[i] = [list(new_left), list(new_right)]
                            del bicliques[j]
                            improved = True
                            break
                
                # Try merge: left = li ∩ lj, right = ri ∪ rj
                if li & lj:
                    new_left = set(li) & set(lj)
                    new_right = set(ri) | set(rj)
                    if new_left and new_right and _is_valid_biclique(new_left, new_right, adj):
                        merge_edges = _biclique_edges_set(new_left, new_right)
                        old_edges = _biclique_edges_set(li, ri) | _biclique_edges_set(lj, rj)
                        if merge_edges == old_edges:
                            bicliques[i] = [list(new_left), list(new_right)]
                            del bicliques[j]
                            improved = True
                            break
            if improved:
                break
    
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    best_result = None
    best_count = float('inf')
    
    # Generate candidate bicliques
    candidates = _generate_candidate_bicliques(adj, all_edges_set, deadline)
    
    # Solve set cover
    if candidates and time.time() < deadline:
        result = _algorithm_x(candidates, all_edges_set, deadline)
        if result:
            if _validate(result, adj, all_edges_set):
                best_result = result
                best_count = len(result)
    
    # Local search improvement
    if best_result and time.time() < deadline:
        improved = _local_search(best_result, adj, all_edges_set, deadline)
        if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
            best_result = improved
            best_count = len(improved)
    
    # Fallback: greedy partition
    if best_result is None or time.time() < deadline - 5:
        remaining = set(all_edges_set)
        bicliques = []
        while remaining and time.time() < deadline:
            e = next(iter(remaining))
            u0, v0 = e
            left, right = _expand_biclique({u0}, {v0}, adj, remaining)
            if not left or not right:
                left, right = {u0}, {v0}
            for u in left:
                for v in right:
                    remaining.discard((min(u, v), max(u, v)))
            bicliques.append((left, right))
        
        if bicliques and _validate(bicliques, adj, all_edges_set):
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_result = bicliques
    
    # Final fallback
    if best_result is None:
        best_result = [[(u,), (v,)] for u, v in P['edges']]
    
    # Convert to proper types
    result = []
    for left, right in best_result:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
