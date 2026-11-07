"""Minimum set cover over maximal bicliques via Bron-Kerbosch + Algorithm-X backtracking."""
import time
import random
import sys

sys.setrecursionlimit(100000)

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _validate(bicliques, adj, all_edges_set):
    n = P['n']
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left).intersection(right):
            return False
        for u in left:
            if u < 0 or u >= n or not isinstance(u, int):
                return False
            for v in right:
                if v < 0 or v >= n or not isinstance(v, int):
                    return False
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    if covered != all_edges_set:
        return False
    return True

def _enumerate_maximal_bicliques(adj, n, deadline):
    """Bron-Kerbosch on the bipartite common-neighbor conflict graph.
    A biclique (L,R) is maximal iff L is a maximal clique in G_L(R) and R is a maximal clique in G_R(L),
    where G_L(R) has vertices V\R and edges (u,v) iff N(u)∩R == N(v)∩R (and u,v both connect to all R).
    We enumerate maximal bicliques by: for each edge (u,v) as seed, expand to a candidate biclique,
    then use a recursive maximal-clique enumeration on the 'common-neighbor' graph to find all maximal extensions.
    To keep it tractable, we enumerate maximal bicliques via a simpler approach:
      - For each subset seed (edge), compute the full common-neighbor closure to get a maximal biclique.
      - Deduplicate.
    This is the standard 'maximal biclique enumeration' via edge-seed expansion.
    """
    bicliques = set()
    
    def expand(left, right):
        """Expand (left, right) to a maximal biclique: add any vertex connected to all of the other side."""
        left = set(left)
        right = set(right)
        changed = True
        while changed:
            changed = False
            # Add to left
            new_left = set()
            for u in range(n):
                if u in left or u in right:
                    continue
                if right.issubset(adj[u]):
                    new_left.add(u)
            if new_left:
                left |= new_left
                changed = True
            # Add to right
            new_right = set()
            for v in range(n):
                if v in left or v in right:
                    continue
                if left.issubset(adj[v]):
                    new_right.add(v)
            if new_right:
                right |= new_right
                changed = True
        right -= left
        left -= right
        return left, right
    
    # Seed with every edge, both orientations
    for u, v in P['edges']:
        if time.time() > deadline:
            break
        l, r = expand({u}, {v})
        if l and r:
            bicliques.add((frozenset(l), frozenset(r)))
        l, r = expand({v}, {u})
        if l and r:
            bicliques.add((frozenset(l), frozenset(r)))
    
    # Also try single-vertex seeds with each neighbor to catch cases
    for u in range(n):
        if time.time() > deadline:
            break
        for v in adj[u]:
            l, r = expand({u}, {v})
            if l and r:
                bicliques.add((frozenset(l), frozenset(r)))
    
    return list(bicliques)

def _biclique_edge_set(left, right):
    s = set()
    for u in left:
        for v in right:
            s.add((min(u, v), max(u, v)))
    return s

def _greedy_set_cover(candidates, all_edges_set, deadline):
    """Greedy set cover: repeatedly pick the biclique covering the most uncovered edges."""
    remaining = set(all_edges_set)
    chosen = []
    # Precompute edge sets
    cand_edges = [_biclique_edge_set(l, r) for l, r in candidates]
    
    while remaining:
        if time.time() > deadline:
            break
        best_idx = -1
        best_count = 0
        for i, (l, r) in enumerate(candidates):
            if not cand_edges[i]:
                continue
            overlap = len(cand_edges[i] & remaining)
            if overlap > best_count:
                best_count = overlap
                best_idx = i
        if best_idx == -1 or best_count == 0:
            # Fallback: cover one edge
            e = next(iter(remaining))
            chosen.append((frozenset({e[0]}), frozenset({e[1]})))
            remaining.discard(e)
        else:
            chosen.append(candidates[best_idx])
            remaining -= cand_edges[best_idx]
    return chosen

def _backtracking_set_cover(candidates, cand_edges, all_edges_set, deadline, best_count):
    """Algorithm-X style backtracking with branch-and-bound.
    Branch on the uncovered edge with the fewest candidate bicliques that cover it.
    """
    # Map each edge to the list of candidate indices that cover it
    edge_to_cands = {}
    for i, es in enumerate(cand_edges):
        for e in es:
            edge_to_cands.setdefault(e, []).append(i)
    
    # Remove candidates that cover no edges
    valid_cands = [i for i in range(len(candidates)) if cand_edges[i]]
    
    # Filter edge_to_cands to only valid candidates
    for e in list(edge_to_cands.keys()):
        edge_to_cands[e] = [i for i in edge_to_cands[e] if i in valid_cands]
    
    best_solution = None
    best_size = best_count  # incumbent from greedy
    
    remaining = set(all_edges_set)
    chosen = []
    
    def solve(remaining, chosen):
        nonlocal best_solution, best_size
        if time.time() > deadline:
            return
        if not remaining:
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_solution = list(chosen)
            return
        # Prune: if current size + 1 >= best_size, no improvement possible
        if len(chosen) + 1 >= best_size:
            return
        
        # Find the uncovered edge with the fewest candidate bicliques
        min_count = float('inf')
        target_edge = None
        target_cands = None
        for e in remaining:
            cands = edge_to_cands.get(e, [])
            if len(cands) < min_count:
                min_count = len(cands)
                target_edge = e
                target_cands = cands
                if min_count == 0:
                    return  # Dead end: edge not covered by any candidate
                if min_count == 1:
                    break  # Can't do better than 1
        
        if target_edge is None:
            return
        
        # Sort candidates by coverage (greedy heuristic within the branch)
        def cov_score(ci):
            return len(cand_edges[ci] & remaining)
        
        sorted_cands = sorted(target_cands, key=cov_score, reverse=True)
        
        for ci in sorted_cands:
            if time.time() > deadline:
                return
            if len(chosen) + 1 >= best_size:
                return
            new_remaining = remaining - cand_edges[ci]
            chosen.append(ci)
            solve(new_remaining, chosen)
            chosen.pop()
    
    solve(remaining, chosen)
    return best_solution

def _randomized_greedy(adj, all_edges_set, deadline, n, seed):
    """Randomized greedy partition for fallback."""
    random.seed(seed)
    remaining = set(all_edges_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        e = random.choice(list(remaining))
        u0, v0 = e
        left_set = {u0}
        right_set = {v0}
        changed = True
        while changed:
            changed = False
            new_left = set()
            for u in range(n):
                if u in left_set or u in right_set:
                    continue
                if right_set.issubset(adj[u]):
                    has_rem = False
                    for v in right_set:
                        if (min(u, v), max(u, v)) in remaining:
                            has_rem = True
                            break
                    if has_rem:
                        new_left.add(u)
                        changed = True
            left_set |= new_left
            new_right = set()
            for v in range(n):
                if v in left_set or v in right_set:
                    continue
                if left_set.issubset(adj[v]):
                    has_rem = False
                    for u in left_set:
                        if (min(u, v), max(u, v)) in remaining:
                            has_rem = True
                            break
                    if has_rem:
                        new_right.add(v)
                        changed = True
            right_set |= new_right
        right_set -= left_set
        left_set -= right_set
        if not left_set or not right_set:
            left_set = {u0}
            right_set = {v0}
        for u in left_set:
            for v in right_set:
                remaining.discard((min(u, v), max(u, v)))
        bicliques.append((left_set, right_set))
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    best_result = None
    best_count = float('inf')
    
    # Phase 1: Enumerate maximal bicliques
    candidates = _enumerate_maximal_bicliques(adj, n, deadline)
    
    if not candidates:
        # Fallback to trivial
        best_result = [(frozenset({u}), frozenset({v})) for u, v in P['edges']]
        best_count = len(best_result)
    
    else:
        # Phase 2: Greedy set cover for initial incumbent
        greedy_result = _greedy_set_cover(candidates, all_edges_set, deadline)
        if greedy_result:
            # Validate
            adj_check = _build_adjacency()
            if _validate([(l, r) for l, r in greedy_result], adj_check, all_edges_set):
                best_result = greedy_result
                best_count = len(greedy_result)
        
        # Phase 3: Exact backtracking set cover with branch-and-bound
        if time.time() < deadline:
            cand_edges = [_biclique_edge_set(l, r) for l, r in candidates]
            bt_result = _backtracking_set_cover(candidates, cand_edges, all_edges_set, deadline, best_count)
            if bt_result is not None:
                bt_bicliques = [candidates[i] for i in bt_result]
                adj_check = _build_adjacency()
                if _validate(bt_bicliques, adj_check, all_edges_set):
                    if len(bt_bicliques) < best_count:
                        best_count = len(bt_bicliques)
                        best_result = bt_bicliques
        
        # Phase 4: Randomized greedy as additional fallback/improvement
        if time.time() < deadline:
            for trial in range(50):
                if time.time() > deadline:
                    break
                rg = _randomized_greedy(adj, all_edges_set, deadline, n, seed=trial)
                if rg and _validate(rg, adj, all_edges_set):
                    if len(rg) < best_count:
                        best_count = len(rg)
                        best_result = rg
    
    # Final fallback
    if best_result is None:
        best_result = [(frozenset({u}), frozenset({v})) for u, v in P['edges']]
    
    # Convert to proper types: list of [list(left), list(right)]
    result = []
    for left, right in best_result:
        result.append([list(left), list(right)])
    
    return result
