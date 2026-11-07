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

def _find_maximal_bicliques(adj, edges_set, n, time_limit=15.0):
    """Find maximal bicliques by growing from each edge using common-neighbor sets."""
    start_time = time.time()
    maximal_bicliques = []
    seen_keys = set()
    adj_set = [set(adj[i]) for i in range(n)]
    edge_list = sorted(edges_set)
    random.shuffle(edge_list)

    for u, v in edge_list:
        if time.time() - start_time > time_limit:
            break

        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            # Expand L: vertices adjacent to all in R
            common_R = adj_set[R[0]].copy() if R else set()
            for r in list(R):
                if r == list(R)[0]:
                    continue
                common_R &= adj_set[r]
            for x in common_R:
                if x not in L and x not in R:
                    L.add(x)
                    changed = True
            # Expand R: vertices adjacent to all in L
            common_L = adj_set[L[0]].copy() if L else set()
            for l in list(L):
                if l == list(L)[0]:
                    continue
                common_L &= adj_set[l]
            for x in common_L:
                if x not in L and x not in R:
                    R.add(x)
                    changed = True

        if not L or not R:
            continue

        # Verify validity
        valid = True
        for a in L:
            for b in R:
                if b not in adj_set[a]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        # Check maximality
        is_max = True
        for x in range(n):
            if x in L or x in R:
                continue
            if all(x in adj_set[r] for r in R):
                is_max = False
                break
            if all(x in adj_set[l] for l in L):
                is_max = False
                break
        if not is_max:
            continue

        key = (tuple(sorted(L)), tuple(sorted(R)))
        key2 = (tuple(sorted(R)), tuple(sorted(L)))
        if key not in seen_keys and key2 not in seen_keys:
            seen_keys.add(key)
            seen_keys.add(key2)
            maximal_bicliques.append((sorted(L), sorted(R)))

    return maximal_bicliques

def _solve_exact_cover(bicliques, edges_set, time_limit=20.0):
    """Solve exact cover using Algorithm X with MRV heuristic."""
    start_time = time.time()

    edge_list = sorted(edges_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    num_edges = len(edge_list)

    biclique_masks = []
    for L, R in bicliques:
        mask = 0
        for u in L:
            for v in R:
                e = tuple(sorted((u, v)))
                if e in edge_to_idx:
                    mask |= (1 << edge_to_idx[e])
        if mask:
            biclique_masks.append(mask)

    if not biclique_masks:
        return None

    all_mask = (1 << num_edges) - 1

    edge_to_bicliques = [[] for _ in range(num_edges)]
    for i, mask in enumerate(biclique_masks):
        for e_idx in range(num_edges):
            if mask & (1 << e_idx):
                edge_to_bicliques[e_idx].append(i)

    best_solution = None
    best_count = float('inf')

    def backtrack(covered_mask, current, depth=0):
        nonlocal best_solution, best_count

        if time.time() - start_time > time_limit:
            return

        if len(current) >= best_count:
            return

        if covered_mask == all_mask:
            if len(current) < best_count:
                best_count = len(current)
                best_solution = list(current)
            return

        min_edge = -1
        min_cands = float('inf')
        for e_idx in range(num_edges):
            if not (covered_mask & (1 << e_idx)):
                cands = [b for b in edge_to_bicliques[e_idx] if (biclique_masks[b] & covered_mask) == 0]
                if len(cands) < min_cands:
                    min_cands = len(cands)
                    min_edge = e_idx
                    if min_cands == 0:
                        return

        if min_cands == 0 or min_edge < 0:
            return

        cands = [b for b in edge_to_bicliques[min_edge] if (biclique_masks[b] & covered_mask) == 0]
        cands.sort(key=lambda b: -biclique_masks[b].bit_count())

        for b in cands:
            current.append(b)
            backtrack(covered_mask | biclique_masks[b], current, depth + 1)
            current.pop()

    backtrack(0, [], 0)

    if best_solution is not None:
        return [bicliques[i] for i in best_solution]
    return None

def _greedy_partition(adj, edges_set, n, time_limit=2.0, seed=42):
    """Greedy biclique partition with randomized anchor selection."""
    start_time = time.time()
    rng = random.Random(seed)
    remaining = set(edges_set)
    bicliques = []

    rem_set = set()
    for u, v in remaining:
        rem_set.add((u, v))
        rem_set.add((v, u))

    while remaining and time.time() - start_time < time_limit:
        deg = [0] * n
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1

        candidates = [v for v in range(n) if deg[v] > 0]
        if not candidates:
            break

        candidates.sort(key=lambda x: (-deg[x], rng.random()))

        best_left = None
        best_right = None
        best_count = 0

        max_anchors = min(len(candidates), 30)

        for anchor in candidates[:max_anchors]:
            if time.time() - start_time > time_limit:
                break

            neighbors = set()
            for v in adj[anchor]:
                if (anchor, v) in rem_set:
                    neighbors.add(v)

            if not neighbors:
                continue

            left = {anchor}
            right = set(neighbors)

            changed = True
            while changed:
                changed = False
                if left:
                    new_right = set(adj[list(left)[0]])
                    for u in left:
                        if u != list(left)[0]:
                            new_right &= adj[u]
                    new_right &= right
                    if new_right != right:
                        right = new_right
                        changed = True
                if right:
                    new_left = set(adj[list(right)[0]])
                    for v in right:
                        if v != list(right)[0]:
                            new_left &= adj[v]
                    new_left &= left
                    if new_left != left:
                        left = new_left
                        changed = True

            if not left or not right:
                continue

            for u in candidates:
                if u in left:
                    continue
                if all(u in adj[v] for v in right):
                    if any((u, v) in rem_set for v in right):
                        left.add(u)
            for v in candidates:
                if v in right:
                    continue
                if all(v in adj[u] for u in left):
                    if any((u, v) in rem_set for u in left):
                        right.add(v)

            if not left or not right:
                continue

            edge_count = 0
            for u in left:
                for v in right:
                    if (u, v) in rem_set:
                        edge_count += 1

            if edge_count > best_count:
                best_count = edge_count
                best_left = sorted(left)
                best_right = sorted(right)

        if best_left is None or best_right is None:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard((u, v))
            rem_set.discard((u, v))
            rem_set.discard((v, u))
            continue

        for u in best_left:
            for v in best_right:
                e = tuple(sorted((u, v)))
                if e in remaining:
                    remaining.discard(e)
                    rem_set.discard((u, v))
                    rem_set.discard((v, u))

        bicliques.append([best_left, best_right])

    for edge in remaining:
        u, v = edge
        bicliques.append([[u], [v]])

    return bicliques

def _validate_bicliques(bicliques, edges_set):
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

def _try_merge(bicliques, edges_set, time_limit=5.0):
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        n_b = len(bicliques)
        for i in range(n_b):
            if time.time() - start_time > time_limit:
                break
            li, ri = bicliques[i][0], bicliques[i][1]
            for j in range(i + 1, n_b):
                if time.time() - start_time > time_limit:
                    break
                lj, rj = bicliques[j][0], bicliques[j][1]

                new_left = set(li) | set(lj)
                new_right = set(ri) | set(rj)

                if new_left & new_right:
                    continue

                valid = True
                for u in new_left:
                    for v in new_right:
                        if tuple(sorted((u, v))) not in edges_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue

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
                    bicliques[i] = [sorted(new_left), sorted(new_right)]
                    del bicliques[j]
                    improved = True
                    break
            if improved:
                break
    return bicliques

def _local_search(bicliques, edges_set, adj, n, time_limit=10.0):
    """Local search: try to reduce biclique count by reassigning edges or merging."""
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        n_b = len(bicliques)
        # Try to absorb a small biclique into a larger one
        for i in range(n_b):
            if time.time() - start_time > time_limit:
                break
            li, ri = bicliques[i][0], bicliques[i][1]
            for j in range(n_b):
                if i == j:
                    continue
                lj, rj = bicliques[j][0], bicliques[j][1]
                # Try adding li to rj (if li vertices connect to all rj)
                if all(u in adj[v] for u in li for v in rj):
                    new_rj = list(set(rj) | set(li))
                    # Check no overlap
                    if not (set(new_rj) & set(lj)):
                        # Verify all edges
                        valid = True
                        for u in lj:
                            for v in new_rj:
                                if tuple(sorted((u, v))) not in edges_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            bicliques[j] = [lj, new_rj]
                            del bicliques[i]
                            improved = True
                            break
                # Try adding ri to lj
                if all(u in adj[v] for u in ri for v in lj):
                    new_lj = list(set(lj) | set(ri))
                    if not (set(new_lj) & set(rj)):
                        valid = True
                        for u in new_lj:
                            for v in rj:
                                if tuple(sorted((u, v))) not in edges_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            bicliques[j] = [new_lj, rj]
                            del bicliques[i]
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 85

    adj = _build_adjacency()
    edges_set = set(tuple(sorted(e)) for e in P['edges'])
    n = P['n']

    def star_partition():
        result = []
        for u in range(n):
            neighbors = sorted(adj[u])
            if neighbors:
                result.append([neighbors, [u]])
        return result

    fallback = star_partition()
    best = fallback
    best_count = len(fallback)

    # Strategy 1: Maximal bicliques + exact cover
    if time.time() < deadline - 35:
        maximal_bicliques = _find_maximal_bicliques(adj, edges_set, n, time_limit=15.0)
        if maximal_bicliques:
            solution = _solve_exact_cover(maximal_bicliques, edges_set, time_limit=20.0)
            if solution is not None and _validate_bicliques(solution, edges_set):
                if len(solution) < best_count:
                    best = solution
                    best_count = len(solution)

    # Strategy 2: Multiple greedy restarts
    if time.time() < deadline - 20:
        for seed in range(30):
            if time.time() > deadline - 10:
                break
            greedy = _greedy_partition(adj, edges_set, n, time_limit=1.5, seed=seed)
            if _validate_bicliques(greedy, edges_set):
                if len(greedy) < best_count:
                    best = greedy
                    best_count = len(greedy)

    # Strategy 3: Merge pass
    if time.time() < deadline - 8:
        merged = _try_merge([list(x) for x in best], edges_set, time_limit=4.0)
        if _validate_bicliques(merged, edges_set) and len(merged) < best_count:
            best = merged
            best_count = len(merged)

    # Strategy 4: Local search
    if time.time() < deadline - 3:
        polished = _local_search([list(x) for x in best], edges_set, adj, n, time_limit=5.0)
        if _validate_bicliques(polished, edges_set) and len(polished) < best_count:
            best = polished
            best_count = len(polished)

    result = []
    for bic in best:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])

    return result

# EVOLVE-BLOCK-END
