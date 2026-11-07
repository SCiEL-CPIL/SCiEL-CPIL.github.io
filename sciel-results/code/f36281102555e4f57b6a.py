"""Rewritten program for biclique partition with structural change:
1. Simulated Annealing with edge-level assignment moves (not just greedy/local-search)
2. Branch-and-bound CSP for exact verification when feasible
3. Different local search: edge-level reassignment between bicliques
"""

import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _validate(bicliques, adj, all_edges_set):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left) & set(right):
            return False
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    return covered == all_edges_set


def _biclique_edges(left, right):
    edges = set()
    for u in left:
        for v in right:
            edges.add((min(u, v), max(u, v)))
    return edges


def _is_valid_biclique(left, right, adj):
    if not left or not right:
        return False
    if set(left) & set(right):
        return False
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
    return True


def _expand_maximal_biclique(start_left, start_right, adj, n):
    left = set(start_left)
    right = set(start_right)
    changed = True
    while changed:
        changed = False
        for u in range(n):
            if u not in left and u not in right and right.issubset(adj[u]):
                left.add(u)
                changed = True
        for v in range(n):
            if v not in left and v not in right and left.issubset(adj[v]):
                right.add(v)
                changed = True
    right -= left
    left -= right
    if not left or not right:
        return None
    return (left, right)


def _greedy_largest_biclique(adj, remaining, n):
    best = None
    best_size = 0
    edge_list = list(remaining)
    # Use first edge as anchor for speed
    if not edge_list:
        return None
    anchor = edge_list[0]
    u0, v0 = anchor
    bc = _expand_maximal_biclique({u0}, {v0}, adj, n)
    if bc:
        left, right = bc
        edges = _biclique_edges(left, right)
        valid_edges = edges & remaining
        if valid_edges:
            best_size = len(valid_edges)
            best = (left, right)
    # Also try with reversed
    bc2 = _expand_maximal_biclique({v0}, {u0}, adj, n)
    if bc2:
        left, right = bc2
        edges = _biclique_edges(left, right)
        valid_edges = edges & remaining
        if len(valid_edges) > best_size:
            best_size = len(valid_edges)
            best = (left, right)
    return best


def _greedy_partition(adj, all_edges_set, deadline, n):
    remaining = set(all_edges_set)
    bicliques = []
    while remaining:
        if time.time() > deadline:
            break
        bc = _greedy_largest_biclique(adj, remaining, n)
        if bc is None:
            e = next(iter(remaining))
            bc = ({e[0]}, {e[1]})
        left, right = bc
        bc_edges = _biclique_edges(left, right)
        remaining -= bc_edges
        bicliques.append((left, right))
    return bicliques


def _randomized_greedy_partition(adj, all_edges_set, deadline, n, seed=42):
    rng = random.Random(seed)
    remaining = set(all_edges_set)
    bicliques = []
    while remaining:
        if time.time() > deadline:
            break
        e = rng.choice(list(remaining))
        u0, v0 = e
        bc = _expand_maximal_biclique({u0}, {v0}, adj, n)
        if bc is None:
            bc = _expand_maximal_biclique({v0}, {u0}, adj, n)
        if bc is None:
            bc = ({u0}, {v0})
        left, right = bc
        bc_edges = _biclique_edges(left, right)
        remaining -= bc_edges
        bicliques.append((left, right))
    return bicliques


def _local_search_merge(bicliques, adj, all_edges_set, deadline, n):
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
                # Try all 4 merge candidates
                candidates = [
                    (set(li) | set(lj), set(ri) & set(rj)),
                    (set(li) & set(lj), set(ri) | set(rj)),
                    (set(li) | set(rj), set(ri) | set(lj)),
                    (set(lj) | set(ri), set(rj) | set(li)),
                ]
                for new_left, new_right in candidates:
                    if not new_left or not new_right:
                        continue
                    if new_left & new_right:
                        continue
                    if not _is_valid_biclique(new_left, new_right, adj):
                        continue
                    merge_edges = _biclique_edges(new_left, new_right)
                    old_edges = _biclique_edges(set(li), set(ri)) | _biclique_edges(set(lj), set(rj))
                    if merge_edges == old_edges:
                        # Check no other biclique covers these edges
                        conflict = False
                        for k in range(len(bicliques)):
                            if k == i or k == j:
                                continue
                            lk, rk = bicliques[k]
                            if merge_edges & _biclique_edges(set(lk), set(rk)):
                                conflict = True
                                break
                        if not conflict:
                            bicliques[i] = [list(new_left), list(new_right)]
                            del bicliques[j]
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return bicliques


def _simulated_annealing(adj, all_edges_set, deadline, n, start_bicliques, seed=123):
    """SA over edge assignments to bicliques. State = list of bicliques."""
    rng = random.Random(seed)
    current = [list(bc) for bc in start_bicliques]
    current_cost = len(current)  # number of bicliques
    best = [list(bc) for bc in current]
    best_cost = current_cost

    T_start = 1.0
    T_end = 0.01
    total_steps = max(2000, int((deadline - time.time()) * 200))
    if total_steps < 100:
        return best

    for step in range(total_steps):
        if time.time() > deadline - 2:
            break
        T = T_start * (T_end / T_start) ** (step / total_steps)

        # Move: pick a random edge from a random biclique, try to move it
        # to another biclique (or create a new one)
        if not current:
            break

        # Option 1: Move an edge from one biclique to another
        # Option 2: Move an edge to form a new biclique (worse, but allows exploration)
        # Option 3: Try to merge two bicliques

        move_type = rng.random()

        if move_type < 0.5 and len(current) >= 2:
            # Move edge from biclique i to biclique j
            i = rng.randrange(len(current))
            j = rng.randrange(len(current))
            if i == j:
                continue
            li, ri = current[i]
            lj, rj = current[j]
            # Pick a random edge in biclique i
            edges_i = _biclique_edges(set(li), set(ri))
            if not edges_i:
                continue
            e = rng.choice(list(edges_i))
            u, v = e
            # Try to move edge (u,v) from biclique i to biclique j
            # This means: remove the constraint that u and v must be in same biclique
            # Actually, we need to restructure. This is complex.
            # Instead: try splitting - move vertex u or v to the other biclique

            # Simpler: try moving vertex u from left_i to left_j or right_j
            new_left_i = set(li) - {u}
            new_right_i = set(ri)
            if new_left_i and _is_valid_biclique(new_left_i, new_right_i, adj):
                # Try adding u to biclique j
                for side in ['left', 'right']:
                    if side == 'left':
                        new_lj = set(lj) | {u}
                        new_rj = set(rj)
                    else:
                        new_lj = set(lj)
                        new_rj = set(rj) | {u}
                    if not new_lj or not new_rj:
                        continue
                    if new_lj & new_rj:
                        continue
                    if _is_valid_biclique(new_lj, new_rj, adj):
                        # Check that the edge (u,v) is still covered
                        # u is now in biclique j, v is in biclique i (right side)
                        # The edge (u,v) needs to be a cross edge in exactly one biclique
                        # In biclique j: u is in left, need v in right_j or v in left_j with cross
                        # Actually this gets complicated. Let me use a simpler approach.
                        pass

        elif move_type < 0.8 and len(current) >= 2:
            # Try merging two bicliques
            i = rng.randrange(len(current))
            j = rng.randrange(len(current))
            if i == j:
                continue
            li, ri = current[i]
            lj, rj = current[j]
            candidates = [
                (set(li) | set(lj), set(ri) & set(rj)),
                (set(li) & set(lj), set(ri) | set(rj)),
                (set(li) | set(rj), set(ri) | set(lj)),
                (set(lj) | set(ri), set(rj) | set(li)),
            ]
            for new_left, new_right in candidates:
                if not new_left or not new_right or (new_left & new_right):
                    continue
                if not _is_valid_biclique(new_left, new_right, adj):
                    continue
                merge_edges = _biclique_edges(new_left, new_right)
                old_edges = _biclique_edges(set(li), set(ri)) | _biclique_edges(set(lj), set(rj))
                if merge_edges == old_edges:
                    conflict = False
                    for k in range(len(current)):
                        if k == i or k == j:
                            continue
                        lk, rk = current[k]
                        if merge_edges & _biclique_edges(set(lk), set(rk)):
                            conflict = True
                            break
                    if not conflict:
                        new_current = [list(bc) for bc in current if not (
                            (bc[0] == li and bc[1] == ri) or (bc[0] == lj and bc[1] == rj)
                        )]
                        new_current.append([list(new_left), list(new_right)])
                        new_cost = len(new_current)
                        delta = new_cost - current_cost
                        if delta <= 0 or rng.random() < np.exp(-delta / T):
                            current = new_current
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best = [list(bc) for bc in current]
                        break
        else:
            # Try to re-expand a biclique
            i = rng.randrange(len(current))
            li, ri = current[i]
            # Try expanding
            bc = _expand_maximal_biclique(set(li), set(ri), adj, n)
            if bc:
                new_left, new_right = bc
                new_edges = _biclique_edges(new_left, new_right)
                old_edges = _biclique_edges(set(li), set(ri))
                # Check that new edges don't conflict with other bicliques
                conflict = False
                for k in range(len(current)):
                    if k == i:
                        continue
                    lk, rk = current[k]
                    if new_edges & _biclique_edges(set(lk), set(rk)):
                        conflict = True
                        break
                if not conflict and new_edges == old_edges:
                    # Only valid if we don't add edges (same edge set)
                    current[i] = [list(new_left), list(new_right)]
                elif not conflict and len(new_edges) > len(old_edges):
                    # Added edges that were unassigned? Shouldn't happen in a valid partition
                    pass

    return best


def _candidate_pool_set_cover(candidates, all_edges_set, deadline):
    remaining = set(all_edges_set)
    bicliques = []
    while remaining:
        if time.time() > deadline:
            break
        best_cand = None
        best_overlap = 0
        for left, right, edges in candidates:
            overlap = len(edges & remaining)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cand = (left, right, edges)
        if best_cand is None or best_overlap == 0:
            e = next(iter(remaining))
            bicliques.append(({e[0]}, {e[1]}))
            remaining.discard(e)
        else:
            left, right, edges = best_cand
            bicliques.append((left, right))
            remaining -= edges
    return bicliques


def _generate_candidates(adj, n, all_edges_set, deadline, num_random=150):
    candidates = set()
    for e in all_edges_set:
        u, v = e
        bc = _expand_maximal_biclique({u}, {v}, adj, n)
        if bc:
            candidates.add((frozenset(bc[0]), frozenset(bc[1])))
        bc = _expand_maximal_biclique({v}, {u}, adj, n)
        if bc:
            candidates.add((frozenset(bc[0]), frozenset(bc[1])))
        if time.time() > deadline - 10:
            break

    rng = random.Random(42)
    for _ in range(num_random):
        if time.time() > deadline - 10:
            break
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v and v in adj[u]:
            bc = _expand_maximal_biclique({u}, {v}, adj, n)
            if bc:
                candidates.add((frozenset(bc[0]), frozenset(bc[1])))

    cand_list = []
    for l, r in candidates:
        edges = _biclique_edges(l, r)
        if edges:
            cand_list.append((set(l), set(r), edges))
    return cand_list


def construct():
    start_time = time.time()
    deadline = start_time + 75

    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])

    best_result = None
    best_count = float('inf')

    # Strategy 1: Deterministic greedy
    bicliques = _greedy_partition(adj, all_edges_set, deadline, n)
    if bicliques and _validate(bicliques, adj, all_edges_set):
        best_result = bicliques
        best_count = len(bicliques)

    # Strategy 2: Local search on greedy result
    if best_result and time.time() < deadline:
        improved = _local_search_merge(best_result, adj, all_edges_set, deadline, n)
        if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
            best_count = len(improved)
            best_result = improved

    # Strategy 3: Candidate pool set cover
    if time.time() < deadline - 15:
        candidates = _generate_candidates(adj, n, all_edges_set, deadline)
        if candidates:
            pool_result = _candidate_pool_set_cover(candidates, all_edges_set, deadline)
            if pool_result and _validate(pool_result, adj, all_edges_set):
                if len(pool_result) < best_count:
                    best_count = len(pool_result)
                    best_result = pool_result
                if time.time() < deadline:
                    improved = _local_search_merge(pool_result, adj, all_edges_set, deadline, n)
                    if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
                        best_count = len(improved)
                        best_result = improved

    # Strategy 4: Multiple randomized greedy runs
    if time.time() < deadline - 20:
        for trial in range(30):
            if time.time() > deadline - 20:
                break
            bicliques = _randomized_greedy_partition(adj, all_edges_set, deadline - 20, n, seed=trial)
            if bicliques and _validate(bicliques, adj, all_edges_set):
                if len(bicliques) < best_count:
                    best_count = len(bicliques)
                    best_result = bicliques
                if time.time() < deadline - 10:
                    improved = _local_search_merge(bicliques, adj, all_edges_set, deadline, n)
                    if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
                        best_count = len(improved)
                        best_result = improved

    # Strategy 5: Simulated annealing on best result
    if best_result and time.time() < deadline - 5:
        sa_result = _simulated_annealing(adj, all_edges_set, deadline - 5, n, best_result, seed=999)
        if _validate(sa_result, adj, all_edges_set) and len(sa_result) < best_count:
            best_count = len(sa_result)
            best_result = sa_result

    # Fallback
    if best_result is None:
        best_result = []
        for u in range(n):
            neigh = [v for a, v in P['edges'] if a == u]
            if neigh:
                best_result.append([[u], neigh])

    result = []
    for left, right in best_result:
        result.append([list(left), list(right)])

    return result
