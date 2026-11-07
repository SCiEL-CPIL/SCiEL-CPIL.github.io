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

def _max_biclique_from_seed(seed_left, seed_right, edge_set, adj, n, deadline):
    left = set(seed_left)
    right = set(seed_right)
    if not left or not right or (left & right):
        return None
    for u in left:
        for v in right:
            if _ek((u, v)) not in edge_set:
                return None
    
    # Initialize candidate sets
    cand_left = set()
    for v in right:
        cand_left.update(adj[v])
    cand_left -= left
    cand_left -= right
    
    cand_right = set()
    for u in left:
        cand_right.update(adj[u])
    cand_right -= left
    cand_right -= right

    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to add vertices to left
        for w in list(cand_left):
            if right.issubset(adj[w]):
                left.add(w)
                cand_left.discard(w)
                changed = True
                # Update cand_right: any future right vertex must be connected to w
                cand_right &= adj[w]
                cand_right -= left
                cand_right -= right
        
        # Try to add vertices to right
        for w in list(cand_right):
            if left.issubset(adj[w]):
                right.add(w)
                cand_right.discard(w)
                changed = True
                # Update cand_left: any future left vertex must be connected to w
                cand_left &= adj[w]
                cand_left -= left
                cand_left -= right

    return (left, right)

def _cross_edges_of_biclique(left, right, edge_set):
    es = set()
    for u in left:
        for v in right:
            k = _ek((u, v))
            if k in edge_set:
                es.add(k)
    return es

def _greedy_partition(edge_set, adj, n, deadline, rng, max_seeds=200):
    remaining = set(edge_set)
    bicliques = []
    if not remaining:
        return bicliques
    
    while remaining and time.time() < deadline:
        rem_edges = list(remaining)
        if not rem_edges:
            break
            
        cur_deg = {v: 0 for v in range(n)}
        for u, v in rem_edges:
            cur_deg[u] += 1
            cur_deg[v] += 1
            
        candidates = []
        for u, v in rem_edges:
            candidates.append((cur_deg[u] + cur_deg[v], u, v))
        candidates.sort(key=lambda x: -x[0])
        
        best_bc = None
        best_cover = 0
        
        # Try top seeds
        top_k = min(max_seeds, len(candidates))
        for idx in range(top_k):
            if time.time() > deadline:
                break
            _, u, v = candidates[idx]
            # Try both orientations
            for left_seed, right_seed in [({u}, {v}), ({v}, {u})]:
                bc = _max_biclique_from_seed(left_seed, right_seed, remaining, adj, n, deadline)
                if bc is None:
                    continue
                l, r = bc
                # Count how many remaining edges this biclique covers
                cover = 0
                for a in l:
                    for b in r:
                        k = _ek((a, b))
                        if k in remaining:
                            cover += 1
                if cover > best_cover:
                    best_cover = cover
                    best_bc = bc
                    if best_cover == len(remaining):
                        break # Perfect cover
            if best_cover == len(remaining):
                break
                
        if best_bc is None or best_cover == 0:
            # Fallback: pick a random remaining edge
            u, v = rem_edges[0]
            best_bc = ({u}, {v})
            best_cover = 1
            
        left, right = best_bc
        # Remove covered edges
        ce = set()
        for a in left:
            for b in right:
                k = _ek((a, b))
                if k in remaining:
                    ce.add(k)
        bicliques.append((left, right))
        remaining -= ce
    return bicliques

def _try_merge(bicliques, edge_set, deadline):
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
                    valid = True
                    for u in nl:
                        for v in nr:
                            if _ek((u, v)) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    merged_cross = set()
                    for u in nl:
                        for v in nr:
                            merged_cross.add(_ek((u, v)))
                    cross_i = set()
                    for u in li:
                        for v in ri:
                            cross_i.add(_ek((u, v)))
                    cross_j = set()
                    for u in lj:
                        for v in rj:
                            cross_j.add(_ek((u, v)))
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

def _local_improve(bicliques, edge_set, adj, n, deadline, max_iters=5000):
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
                for v in list(li):
                    new_li = li - {v}
                    new_rj = rj | {v}
                    if not new_li:
                        continue
                    if new_rj & new_li:
                        continue
                    if not lj.issubset(adj[v]):
                        continue
                    if not ri.issubset(adj[v]) or not all(adj[a] & ri == ri for a in new_li):
                        continue
                    if not all(adj[a] & new_rj == new_rj for a in lj):
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
            if improved:
                break
        if not improved:
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    return bicliques
                li, ri = bicliques[i]
                for j in range(len(bicliques)):
                    if i == j:
                        continue
                    lj, rj = bicliques[j]
                    for v in list(li):
                        for w in list(rj):
                            new_li = (li - {v}) | {w}
                            new_rj = (rj - {w}) | {v}
                            if not new_li or not new_rj:
                                continue
                            if new_li & new_rj:
                                continue
                            if new_li & li == new_li and new_rj & rj == new_rj:
                                continue
                            if not all(adj[a] & ri == ri for a in new_li):
                                continue
                            if not all(adj[a] & new_rj == new_rj for a in lj):
                                continue
                            cross_i_new = set(_ek((a, b)) for a in new_li for b in ri)
                            cross_j_new = set(_ek((a, b)) for a in lj for b in new_rj)
                            if cross_i_new & cross_j_new:
                                continue
                            bicliques[i] = (new_li, ri)
                            bicliques[j] = (lj, new_rj)
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        bicliques = _try_merge(bicliques, edge_set, deadline)
    return bicliques

def _edge_reassign_polish(bicliques, edge_set, adj, n, deadline):
    bicliques = list(bicliques)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        sizes = [len(l) * len(r) for l, r in bicliques]
        order = sorted(range(len(bicliques)), key=lambda i: sizes[i])
        for si in order:
            if time.time() > deadline:
                break
            if si >= len(bicliques):
                continue
            li, ri = bicliques[si]
            bc_edges = set(_ek((u, v)) for u in li for v in ri)
            absorbed = False
            for sj in range(len(bicliques)):
                if sj == si:
                    continue
                lj, rj = bicliques[sj]
                for nl, nr in [(lj | li, rj), (lj, rj | ri), (lj | ri, rj | li), (lj, rj | li)]:
                    if not nl or not nr or (nl & nr):
                        continue
                    valid = True
                    for u in nl:
                        for v in nr:
                            if _ek((u, v)) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    new_cross = set(_ek((u, v)) for u in nl for v in nr)
                    if not bc_edges.issubset(new_cross):
                        continue
                    overlap = False
                    for sk in range(len(bicliques)):
                        if sk == si or sk == sj:
                            continue
                        lk, rk = bicliques[sk]
                        ck = set(_ek((u, v)) for u in lk for v in rk)
                        if new_cross & ck:
                            overlap = True
                            break
                    if overlap:
                        continue
                    bicliques[sj] = (nl, nr)
                    bicliques.pop(si)
                    improved = True
                    absorbed = True
                    break
                if absorbed:
                    break
            if absorbed:
                break
    return bicliques

def _gf2_rank_and_bicliques(n, edges, deadline):
    """
    Use GF(2) linear algebra to find a small biclique partition.
    The idea: each biclique corresponds to a rank-1 matrix u*v^T over GF(2).
    We want to express the adjacency matrix as a sum of rank-1 matrices.
    The minimum number is the rank over GF(2).
    
    We compute the rank and then extract the rank-1 factors.
    """
    edge_set = set(_ek(e) for e in edges)
    
    # Build adjacency matrix over GF(2)
    # A[i][j] = 1 if (i,j) is an edge
    A = [[0]*n for _ in range(n)]
    for u, v in edges:
        A[u][v] = 1
        A[v][u] = 1
    
    # Compute rank over GF(2) using Gaussian elimination
    # Work on a copy
    M = [row[:] for row in A]
    
    rank = 0
    pivot_cols = []
    for col in range(n):
        # Find pivot row
        pivot_row = -1
        for row in range(rank, n):
            if M[row][col] == 1:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        # Swap
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        # Eliminate
        for row in range(n):
            if row != rank and M[row][col] == 1:
                for c in range(col, n):
                    M[row][c] ^= M[rank][c]
        rank += 1
        pivot_cols.append(col)
    
    # Now we have rank r. We need to express A as sum of r rank-1 matrices.
    # The pivot rows form a basis. Let's extract the rank-1 factors.
    # Actually, we need to be more careful. Let's use a different approach.
    
    # Find all maximal bicliques and use a greedy set cover, but guided by the rank.
    # The rank gives us a lower bound on the number of bicliques needed.
    
    adj = _adj(n, edges)
    
    # Find maximal bicliques
    all_bicliques = []
    seen = set()
    
    for i in range(n):
        for j in range(i+1, n):
            if _ek((i,j)) in edge_set:
                bc = _max_biclique_from_seed({i}, {j}, edge_set, adj, n, deadline)
                if bc:
                    l, r = bc
                    key = (frozenset(l), frozenset(r))
                    if key not in seen:
                        seen.add(key)
                        all_bicliques.append((l, r))
                bc2 = _max_biclique_from_seed({j}, {i}, edge_set, adj, n, deadline)
                if bc2:
                    l, r = bc2
                    key = (frozenset(l), frozenset(r))
                    if key not in seen:
                        seen.add(key)
                        all_bicliques.append((l, r))
    
    # Greedy set cover
    remaining = set(edge_set)
    bicliques_used = []
    
    while remaining and time.time() < deadline:
        best_bc = None
        best_count = 0
        for l, r in all_bicliques:
            ce = set()
            for u in l:
                for v in r:
                    k = _ek((u, v))
                    if k in remaining:
                        ce.add(k)
            if len(ce) > best_count:
                best_count = len(ce)
                best_bc = (l, r)
        if best_bc is None or best_count == 0:
            u, v = next(iter(remaining))
            bicliques_used.append(({u}, {v}))
            remaining.discard(_ek((u, v)))
            continue
        l, r = best_bc
        ce = set()
        for u in l:
            for v in r:
                k = _ek((u, v))
                if k in remaining:
                    ce.add(k)
        bicliques_used.append((l, r))
        remaining -= ce
    
    return bicliques_used, rank

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    all_edge_set = set(_ek(e) for e in edges)
    adj = _adj(n, edges)
    
    rng = random.Random(42)
    
    best_bicliques = None
    best_count = float('inf')
    
    # Strategy 1: GF(2)-guided approach
    bc_gf2, gf2_rank = _gf2_rank_and_bicliques(n, edges, deadline)
    bc_gf2 = _try_merge(bc_gf2, all_edge_set, deadline)
    bc_gf2 = _local_improve(bc_gf2, all_edge_set, adj, n, deadline)
    bc_gf2 = _edge_reassign_polish(bc_gf2, all_edge_set, adj, n, deadline)
    if _validate(bc_gf2, all_edge_set):
        best_bicliques = bc_gf2
        best_count = len(bc_gf2)
    
    # Strategy 2: Standard greedy with restarts
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        bc = _greedy_partition(all_edge_set, adj, n, deadline, rng, max_seeds=150)
        bc = _try_merge(bc, all_edge_set, deadline)
        bc = _local_improve(bc, all_edge_set, adj, n, deadline)
        bc = _edge_reassign_polish(bc, all_edge_set, adj, n, deadline)
        if _validate(bc, all_edge_set):
            if len(bc) < best_count:
                best_bicliques = bc
                best_count = len(bc)
    
    # Strategy 3: More restarts if time permits
    remaining_time = deadline - time.time()
    n_restarts = int(min(15, max(0, remaining_time / 3)))
    
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        rng = random.Random(42 + r * 137)
        max_seeds = 80 + r * 15
        bc = _greedy_partition(all_edge_set, adj, n, deadline, rng, max_seeds=max_seeds)
        bc = _try_merge(bc, all_edge_set, deadline)
        bc = _local_improve(bc, all_edge_set, adj, n, deadline)
        bc = _edge_reassign_polish(bc, all_edge_set, adj, n, deadline)
        if _validate(bc, all_edge_set):
            if len(bc) < best_count:
                best_bicliques = bc
                best_count = len(bc)
    
    if best_bicliques is None:
        best_bicliques = [({u}, {v}) for u, v in edges]
    
    # Final validation
    if not _validate(best_bicliques, all_edge_set):
        best_bicliques = [({u}, {v}) for u, v in edges]
    
    # Convert to proper format
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
