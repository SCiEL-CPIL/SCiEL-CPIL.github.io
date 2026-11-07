import time
import random
from collections import deque

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _ek(e):
    a, b = e
    return (a, b) if a < b else (b, a)

def _adj(n, edges):
    A = [set() for _ in range(n)]
    for u, v in edges:
        A[u].add(v)
        A[v].add(u)
    return A

def _cross_edges(left, right, edge_set):
    es = set()
    for u in left:
        for v in right:
            k = _ek((u, v))
            if k in edge_set:
                es.add(k)
    return es

def _grow_biclique(seed_l, seed_r, edge_set, n, deadline):
    left = set(seed_l)
    right = set(seed_r)
    if not left or not right or (left & right):
        return None
    for u in left:
        for v in right:
            if _ek((u, v)) not in edge_set:
                return None
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for w in range(n):
            if w in left or w in right:
                continue
            if all(_ek((w, r)) in edge_set for r in right):
                left.add(w)
                changed = True
        for w in range(n):
            if w in left or w in right:
                continue
            if all(_ek((l, w)) in edge_set for l in left):
                right.add(w)
                changed = True
    return (left, right)

def _greedy_partition(edge_set, n, deadline, rng, max_seeds=50):
    remaining = set(edge_set)
    bicliques = []
    if not remaining:
        return bicliques
    while remaining and time.time() < deadline:
        rem_edges = list(remaining)
        cur_deg = {v: 0 for v in range(n)}
        for u, v in rem_edges:
            cur_deg[u] += 1
            cur_deg[v] += 1
        candidates = []
        for u, v in rem_edges:
            candidates.append((cur_deg[u] + cur_deg[v], u, v))
        candidates.sort(key=lambda x: -x[0])
        best_bc = None
        best_size = 0
        top_k = min(max_seeds, len(candidates))
        for idx in range(top_k):
            if time.time() > deadline:
                break
            _, u, v = candidates[idx]
            for ls, rs in [({u}, {v}), ({v}, {u})]:
                bc = _grow_biclique(ls, rs, remaining, n, deadline)
                if bc is None:
                    continue
                l, r = bc
                size = len(l) * len(r)
                if size > best_size:
                    best_size = size
                    best_bc = bc
        if best_bc is None:
            u, v = rem_edges[0]
            best_bc = ({u}, {v})
        left, right = best_bc
        ce = _cross_edges(left, right, remaining)
        if not ce:
            u, v = rem_edges[0]
            bicliques.append(({u}, {v}))
            remaining.discard(_ek((u, v)))
            continue
        bicliques.append((left, right))
        remaining -= ce
    return bicliques

def _is_valid_biclique(left, right, edge_set):
    if not left or not right or (left & right):
        return False
    for u in left:
        for v in right:
            if _ek((u, v)) not in edge_set:
                return False
    return True

def _try_merge(bicliques, edge_set, n, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        bicliques = list(bicliques)
        n_bc = len(bicliques)
        for i in range(n_bc):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_bc):
                if time.time() > deadline:
                    break
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                for combo in [(li | lj, ri | rj), (li | rj, ri | lj), (ri | lj, li | rj), (ri | rj, li | lj)]:
                    nl, nr = combo
                    if not nl or not nr or (nl & nr):
                        continue
                    if not _is_valid_biclique(nl, nr, edge_set):
                        continue
                    merged_cross = _cross_edges(nl, nr, edge_set)
                    cross_i = _cross_edges(li, ri, edge_set)
                    cross_j = _cross_edges(lj, rj, edge_set)
                    if merged_cross == (cross_i | cross_j):
                        new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
                        new_bicliques.append((nl, nr))
                        bicliques = new_bicliques
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return bicliques

def _remove_redundant(bicliques, edge_set, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_bc = len(bicliques)
        for i in range(n_bc):
            if time.time() > deadline:
                break
            e_i = _cross_edges(bicliques[i][0], bicliques[i][1], edge_set)
            covered = set()
            for j in range(n_bc):
                if i == j:
                    continue
                e_j = _cross_edges(bicliques[j][0], bicliques[j][1], edge_set)
                covered |= e_j
            if e_i.issubset(covered):
                bicliques.pop(i)
                improved = True
                break
    return bicliques

def _validate_partition(bicliques, all_edge_set):
    covered = set()
    for l, r in bicliques:
        if not l or not r or (l & r):
            return False
        e = _cross_edges(l, r, all_edge_set)
        if not e.issubset(all_edge_set):
            return False
        if e & covered:
            return False
        covered |= e
    return covered == all_edge_set

def construct():
    start_time = time.time()
    deadline = start_time + 82
    n = P['n']
    edges = P['edges']
    all_edge_set = frozenset(_ek(e) for e in edges)
    
    best_partition = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Multiple randomized restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        # Vary the random seed and parameters for diversity
        rng.seed(restart * 1000 + 42)
        
        # Try different max_seeds values
        max_seeds = 30 + (restart % 3) * 20
        
        partition = _greedy_partition(all_edge_set, n, deadline, rng, max_seeds)
        
        if not partition:
            continue
        
        # Refinement: remove redundant
        partition = _remove_redundant(partition, all_edge_set, deadline)
        
        # Refinement: try merges
        partition = _try_merge(partition, all_edge_set, n, deadline)
        
        # Remove redundant again after merges
        partition = _remove_redundant(partition, all_edge_set, deadline)
        
        # Validate
        if _validate_partition(partition, all_edge_set):
            count = len(partition)
            if count < best_count:
                best_count = count
                best_partition = partition
    
    # If no valid partition found, use fallback
    if best_partition is None:
        best_partition = [({u}, {v}) for u, v in edges]
    
    # Convert to output format with proper types
    result = []
    for l, r in best_partition:
        result.append([list(l), list(r)])
    
    return result
