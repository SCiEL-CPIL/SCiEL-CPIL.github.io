"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

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

def _build_adj_matrix(n, edge_set):
    """Build boolean adjacency matrix as list of bytearrays for fast membership."""
    M = [bytearray(n) for _ in range(n)]
    for u, v in edge_set:
        M[u][v] = 1
        M[v][u] = 1
    return M

def _is_complete_bipartite(left, right, adj_mat, n):
    """Check if all cross edges exist using adjacency matrix."""
    for u in left:
        row = adj_mat[u]
        for v in right:
            if not row[v]:
                return False
    return True

def _max_biclique_from_seed(seed_left, seed_right, edge_set, adj_mat, n, deadline):
    """Grow a biclique from initial left/right sets using adjacency matrix for speed."""
    left = set(seed_left)
    right = set(seed_right)
    if not left or not right or (left & right):
        return None
    # Verify all cross edges exist
    if not _is_complete_bipartite(left, right, adj_mat, n):
        return None
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to add to left: vertex w connects to all in right
        for w in range(n):
            if w in left or w in right:
                continue
            if all(adj_mat[w][r] for r in right):
                left.add(w)
                changed = True
        # Try to add to right: vertex w connects to all in left
        for w in range(n):
            if w in left or w in right:
                continue
            if all(adj_mat[l][w] for l in left):
                right.add(w)
                changed = True
    return (left, right)

def _cross_edges_of_biclique(left, right, edge_set):
    es = set()
    for u in left:
        for v in right:
            k = _ek((u, v))
            if k in edge_set:
                es.add(k)
    return es

def _greedy_partition(edge_set, adj_mat, n, deadline, rng, max_seeds=200):
    """Greedy: repeatedly pick a maximal biclique, remove its edges."""
    remaining = set(edge_set)
    bicliques = []
    
    if not remaining:
        return bicliques
    
    while remaining and time.time() < deadline:
        rem_edges = list(remaining)
        
        # Compute current degrees
        cur_deg = {v: 0 for v in range(n)}
        for u, v in rem_edges:
            cur_deg[u] += 1
            cur_deg[v] += 1
        
        # Candidate seeds: edges in remaining, sorted by combined degree
        candidates = []
        for u, v in rem_edges:
            candidates.append((cur_deg[u] + cur_deg[v], u, v))
        candidates.sort(key=lambda x: -x[0])
        
        # Try top seeds
        best_bc = None
        best_size = 0
        top_k = min(max_seeds, len(candidates))
        
        for idx in range(top_k):
            if time.time() > deadline:
                break
            _, u, v = candidates[idx]
            # Try both orientations
            for left_seed, right_seed in [({u}, {v}), ({v}, {u})]:
                bc = _max_biclique_from_seed(left_seed, right_seed, remaining, adj_mat, n, deadline)
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
        ce = _cross_edges_of_biclique(left, right, remaining)
        if not ce:
            u, v = rem_edges[0]
            bicliques.append(({u}, {v}))
            remaining.discard(_ek((u, v)))
            continue
        
        bicliques.append((left, right))
        remaining -= ce
    
    return bicliques

def _try_merge(bicliques, edge_set, adj_mat, n, deadline):
    """Try to merge pairs of bicliques if compatible."""
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
                
                # Try 4 combinations
                for combo in [(li | lj, ri | rj), (li | rj, ri | lj), (ri | lj, li | rj), (ri | rj, li | lj)]:
                    nl, nr = combo
                    if not nl or not nr or (nl & nr):
                        continue
                    if not _is_complete_bipartite(nl, nr, adj_mat, n):
                        continue
                    # Check cross edges of merged == union of cross edges
                    merged_cross = set(_ek((u, v)) for u in nl for v in nr)
                    cross_i = set(_ek((u, v)) for u in li for v in ri)
                    cross_j = set(_ek((u, v)) for u in lj for v in rj)
                    if merged_cross == cross_i | cross_j:
                        bicliques[i] = (nl, nr)
                        bicliques.pop(j)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return bicliques

def _validate(bicliques, all_edge_set):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if left & right:
            return False
        for u in left:
            for v in right:
                k = _ek((u, v))
                if k not in all_edge_set:
                    return False
                if k in covered:
                    return False
                covered.add(k)
    return covered == all_edge_set

def _local_improve(bicliques, edge_set, adj_mat, n, deadline, max_iters=5000):
    """Local search: try moves and swaps to reduce biclique count."""
    bicliques = list(bicliques)
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        for i in range(len(bicliques)):
            if time.time() > deadline:
                return bicliques
            li, ri = bicliques[i]
            for j in range(len(bicliques)):
                if i == j:
                    continue
                lj, rj = bicliques[j]
                
                # Try moving a vertex from li to rj
                for v in list(li):
                    new_li = li - {v}
                    new_rj = rj | {v}
                    if not new_li or (new_rj & new_li):
                        continue
                    if not all(adj_mat[v][w] for w in lj):
                        continue
                    if not _is_complete_bipartite(new_li, ri, adj_mat, n):
                        continue
                    if not _is_complete_bipartite(lj, new_rj, adj_mat, n):
                        continue
                    cross_i_new = set(_ek((a, b)) for a in new_li for b in ri)
                    cross_j_new = set(_ek((a, b)) for a in lj for b in new_rj)
                    if cross_i_new & cross_j_new:
                        continue
                    bicliques[i] = (new_li, ri)
                    bicliques[j] = (lj, new_rj)
                    if not new_li:
                        bicliques.pop(i)
                        if j > i:
                            j -= 1
                        if j >= len(bicliques):
                            break
                    improved = True
                    break
                if improved:
                    break
                
                # Try swapping a vertex between li and lj
                for v in list(li):
                    for w in list(lj):
                        if v == w:
                            continue
                        new_li = (li - {v}) | {w}
                        new_lj = (lj - {w}) | {v}
                        if not new_li or not new_lj:
                            continue
                        if (new_li & ri) or (new_lj & rj):
                            continue
                        if not _is_complete_bipartite(new_li, ri, adj_mat, n):
                            continue
                        if not _is_complete_bipartite(new_lj, rj, adj_mat, n):
                            continue
                        cross_i_new = set(_ek((a, b)) for a in new_li for b in ri)
                        cross_j_new = set(_ek((a, b)) for a in new_lj for b in rj)
                        if cross_i_new & cross_j_new:
                            continue
                        bicliques[i] = (new_li, ri)
                        bicliques[j] = (new_lj, rj)
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        # Also try merging
        bicliques = _try_merge(bicliques, edge_set, adj_mat, n, deadline)
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    edges = P['edges']
    all_edge_set = set(_ek(e) for e in edges)
    adj_mat = _build_adj_matrix(n, all_edge_set)
    
    best_bicliques = None
    best_count = float('inf')
    
    # Deterministic greedy
    bc = _greedy_partition(all_edge_set, adj_mat, n, deadline, random.Random(42), max_seeds=100)
    bc = _try_merge(bc, all_edge_set, adj_mat, n, deadline)
    if _validate(bc, all_edge_set):
        best_bicliques = bc
        best_count = len(bc)
    
    # Random restarts
    remaining_time_budget = deadline - time.time()
    n_restarts = int(min(30, max(0, remaining_time_budget / 2)))
    
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        bc = _greedy_partition(all_edge_set, adj_mat, n, deadline, random.Random(42 + r * 137), max_seeds=50 + r * 5)
        bc = _try_merge(bc, all_edge_set, adj_mat, n, deadline)
        if _validate(bc, all_edge_set):
            if len(bc) < best_count:
                best_bicliques = bc
                best_count = len(bc)
    
    # Local improvement on best
    if best_bicliques is not None:
        best_bicliques = _local_improve(best_bicliques, all_edge_set, adj_mat, n, deadline)
        if not _validate(best_bicliques, all_edge_set):
            bc2 = _greedy_partition(all_edge_set, adj_mat, n, deadline, random.Random(42), max_seeds=100)
            bc2 = _try_merge(bc2, all_edge_set, adj_mat, n, deadline)
            if _validate(bc2, all_edge_set):
                best_bicliques = bc2
            else:
                best_bicliques = [({u}, {v}) for u, v in edges]
    
    if best_bicliques is None:
        best_bicliques = [({u}, {v}) for u, v in edges]
    
    if not _validate(best_bicliques, all_edge_set):
        best_bicliques = [({u}, {v}) for u, v in edges]
    
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
