"""Edge-weighted greedy biclique partition with local search (swap/merge/split)."""
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


def _biclique_edges(left, right):
    """Return set of edges in biclique (left, right)."""
    s = set()
    for u in left:
        for v in right:
            s.add((min(u, v), max(u, v)))
    return s


def _expand_biclique(left, right, adj, n, only_uncovered=None):
    """Expand (left, right) to a maximal biclique. If only_uncovered is a set of edges,
    only add vertices that contribute at least one uncovered edge."""
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
                if only_uncovered is None:
                    new_left.add(u)
                else:
                    for v in right:
                        e = (min(u, v), max(u, v))
                        if e in only_uncovered:
                            new_left.add(u)
                            break
        if new_left:
            left |= new_left
            changed = True
        # Add to right
        new_right = set()
        for v in range(n):
            if v in left or v in right:
                continue
            if left.issubset(adj[v]):
                if only_uncovered is None:
                    new_right.add(v)
                else:
                    for u in left:
                        e = (min(u, v), max(u, v))
                        if e in only_uncovered:
                            new_right.add(v)
                            break
        if new_right:
            right |= new_right
            changed = True
    right -= left
    left -= right
    return left, right


def _greedy_partition(adj, all_edges_set, n, deadline, seed=0):
    """Greedy: repeatedly pick the largest biclique (by edge count) from remaining edges."""
    rng = random.Random(seed)
    remaining = set(all_edges_set)
    bicliques = []

    while remaining:
        if time.time() > deadline:
            break
        # Find best biclique: try each edge as seed, expand
        best_biclique = None
        best_count = 0
        # Sample edges if too many
        edges_list = list(remaining)
        if len(edges_list) > 200:
            edges_list = rng.sample(edges_list, 200)
        for e in edges_list:
            u, v = e
            l, r = _expand_biclique({u}, {v}, adj, n, only_uncovered=remaining)
            if l and r:
                b_edges = _biclique_edges(l, r)
                # Only count edges that are in remaining
                valid_edges = b_edges & remaining
                if len(valid_edges) > best_count:
                    best_count = len(valid_edges)
                    best_biclique = (l, r, valid_edges)
        if best_biclique is None:
            # Fallback: single edge
            e = remaining.pop()
            bicliques.append(({e[0]}, {e[1]}))
        else:
            l, r, valid_edges = best_biclique
            remaining -= valid_edges
            bicliques.append((l, r))
    return bicliques


def _local_search(bicliques, adj, all_edges_set, n, deadline, max_rounds=200):
    """Local search with merge, split, and re-assign moves."""
    if not bicliques:
        return bicliques

    # Work with mutable lists
    bicliques = [(set(l), set(r)) for l, r in bicliques]

    def get_edges(bi):
        return _biclique_edges(bi[0], bi[1])

    def total_edges(bicliques_list):
        return sum(len(get_edges(bi)) for bi in bicliques_list)

    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        if time.time() > deadline:
            break
        improved = False
        rounds += 1

        # Move 1: Merge two bicliques if their union is a valid biclique
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(bicliques)):
                li, ri = bicliques[i]
                lj, rj = bicliques[j]
                # Try merging: L = li ∪ lj, R = ri ∩ rj  (if all L connect to all R)
                # Or L = li ∩ lj, R = ri ∪ rj
                # Or L = li ∪ lj, R = ri ∪ rj (check validity)
                # Or L = li, R = ri ∪ rj (if valid)
                # Or L = li ∪ lj, R = ri (if valid)
                candidates = []

                # Candidate: L = li ∪ lj, R = ri ∩ rj
                L = li | lj
                R = ri & rj
                if L and R and not (L & R):
                    valid = True
                    for u in L:
                        if not R.issubset(adj[u]):
                            valid = False
                            break
                    if valid:
                        candidates.append((L, R))

                # Candidate: L = li ∩ lj, R = ri ∪ rj
                L = li & lj
                R = ri | rj
                if L and R and not (L & R):
                    valid = True
                    for u in L:
                        if not R.issubset(adj[u]):
                            valid = False
                            break
                    if valid:
                        candidates.append((L, R))

                # Candidate: L = li ∪ lj, R = ri
                L = li | lj
                R = ri
                if L and R and not (L & R):
                    valid = True
                    for u in L:
                        if not R.issubset(adj[u]):
                            valid = False
                            break
                    if valid:
                        candidates.append((L, R))

                # Candidate: L = li, R = ri ∪ rj
                L = li
                R = ri | rj
                if L and R and not (L & R):
                    valid = True
                    for u in L:
                        if not R.issubset(adj[u]):
                            valid = False
                            break
                    if valid:
                        candidates.append((L, R))

                # Candidate: L = li ∪ lj, R = ri ∪ rj
                L = li | lj
                R = ri | rj
                if L and R and not (L & R):
                    valid = True
                    for u in L:
                        if not R.issubset(adj[u]):
                            valid = False
                            break
                    if valid:
                        candidates.append((L, R))

                # Candidate: L = li, R = ri ∩ rj
                L = li
                R = ri & rj
                if L and R and not (L & R):
                    valid = True
                    for u in L:
                        if not R.issubset(adj[u]):
                            valid = False
                            break
                    if valid:
                        candidates.append((L, R))

                # Candidate: L = li ∩ lj, R = ri
                L = li & lj
                R = ri
                if L and R and not (L & R):
                    valid = True
                    for u in L:
                        if not R.issubset(adj[u]):
                            valid = False
                            break
                    if valid:
                        candidates.append((L, R))

                # Candidate: L = li, R = rj
                L = li
                R = rj
                if L and R and not (L & R):
                    valid = True
                    for u in L:
                        if not R.issubset(adj[u]):
                            valid = False
                            break
                    if valid:
                        candidates.append((L, R))

                # Candidate: L = lj, R = ri
                L = lj
                R = ri
                if L and R and not (L & R):
                    valid = True
                    for u in L:
                        if not R.issubset(adj[u]):
                            valid = False
                            break
                    if valid:
                        candidates.append((L, R))

                for L, R in candidates:
                    new_bicliques = [bicliques[k] for k in range(len(bicliques)) if k not in (i, j)]
                    new_bicliques.append((L, R))
                    # Check coverage
                    covered = set()
                    valid_cover = True
                    for bl, br in new_bicliques:
                        for u in bl:
                            for v in br:
                                e = (min(u, v), max(u, v))
                                if e in covered:
                                    valid_cover = False
                                    break
                                covered.add(e)
                        if not valid_cover:
                            break
                    if valid_cover and covered == all_edges_set:
                        bicliques = new_bicliques
                        improved = True
                        break
            if improved:
                break
        if improved:
            continue

        # Move 2: Split a biclique into two if it reduces total count
        # (only useful if we can merge elsewhere, so skip for now)

        # Move 3: Re-assign a vertex from one side to the other side of another biclique
        # This is complex; try simple: for each biclique, try moving a vertex to another biclique
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            li, ri = bicliques[i]
            for u in list(li):
                # Try moving u to right side of some other biclique
                for j in range(len(bicliques)):
                    if j == i:
                        continue
                    lj, rj = bicliques[j]
                    # New biclique i: (li - {u}, ri), new biclique j: (lj, rj ∪ {u})
                    new_li = li - {u}
                    new_rj = rj | {u}
                    if not new_li or not new_rj:
                        continue
                    if new_li & ri:
                        continue
                    if new_li & new_rj:
                        continue
                    # Check validity of new biclique i
                    valid_i = True
                    for uu in new_li:
                        if not ri.issubset(adj[uu]):
                            valid_i = False
                            break
                    if not valid_i:
                        continue
                    # Check validity of new biclique j
                    valid_j = True
                    for uu in lj:
                        if not new_rj.issubset(adj[uu]):
                            valid_j = False
                            break
                    if not valid_j:
                        continue
                    # Check no overlap in edges
                    new_bicliques = []
                    for k in range(len(bicliques)):
                        if k == i:
                            new_bicliques.append((new_li, ri))
                        elif k == j:
                            new_bicliques.append((lj, new_rj))
                        else:
                            new_bicliques.append(bicliques[k])
                    covered = set()
                    valid_cover = True
                    for bl, br in new_bicliques:
                        for uu in bl:
                            for vv in br:
                                e = (min(uu, vv), max(uu, vv))
                                if e in covered:
                                    valid_cover = False
                                    break
                                covered.add(e)
                        if not valid_cover:
                            break
                    if valid_cover and covered == all_edges_set:
                        bicliques = new_bicliques
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Move 4: Try to expand each biclique to be maximal
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            li, ri = bicliques[i]
            # Compute edges of other bicliques
            other_edges = set()
            for j in range(len(bicliques)):
                if j != i:
                    other_edges |= _biclique_edges(bicliques[j][0], bicliques[j][1])
            # Expand biclique i using only edges not in other bicliques
            remaining_for_i = all_edges_set - other_edges
            new_li, new_ri = _expand_biclique(li, ri, adj, n, only_uncovered=remaining_for_i)
            if new_li != li or new_ri != ri:
                # Check if expansion is valid (no overlap with others)
                new_edges = _biclique_edges(new_li, new_ri)
                if not (new_edges & other_edges):
                    bicliques[i] = (new_li, new_ri)
                    improved = True
                    break
        if improved:
            continue

        # Move 5: Try to re-partition: remove one biclique, redistribute its edges
        for i in range(len(bicliques)):
            if time.time() > deadline:
                break
            li, ri = bicliques[i]
            bi_edges = _biclique_edges(li, ri)
            remaining = all_edges_set - bi_edges
            other_bicliques = [bicliques[j] for j in range(len(bicliques)) if j != i]
            # Try to cover bi_edges with existing bicliques + new ones
            # Simple: try to add each edge of bi_edges to an existing biclique
            new_bicliques = [set(l) for l, r in other_bicliques]
            new_rights = [set(r) for l, r in other_bicliques]
            # This is complex; skip for now
            pass

    return bicliques


def _randomized_greedy_partition(adj, all_edges_set, n, deadline, seed):
    """Randomized greedy with different edge ordering."""
    rng = random.Random(seed)
    remaining = set(all_edges_set)
    bicliques = []
    edges_list = list(remaining)
    rng.shuffle(edges_list)

    for e in edges_list:
        if time.time() > deadline:
            break
        if e not in remaining:
            continue
        u, v = e
        l, r = _expand_biclique({u}, {v}, adj, n, only_uncovered=remaining)
        if l and r:
            b_edges = _biclique_edges(l, r) & remaining
            if b_edges:
                remaining -= b_edges
                bicliques.append((l, r))
    # Handle any remaining edges
    while remaining:
        if time.time() > deadline:
            break
        e = remaining.pop()
        bicliques.append(({e[0]}, {e[1]}))
    return bicliques


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])

    best_result = None
    best_count = float('inf')

    # Phase 1: Multiple greedy starts with different seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        result = _greedy_partition(adj, all_edges_set, n, deadline, seed=seed)
        if result and _validate(result, adj, all_edges_set):
            if len(result) < best_count:
                best_count = len(result)
                best_result = result

    # Phase 2: Randomized greedy with different orderings
    for seed in range(30):
        if time.time() > deadline:
            break
        result = _randomized_greedy_partition(adj, all_edges_set, n, deadline, seed=seed)
        if result and _validate(result, adj, all_edges_set):
            if len(result) < best_count:
                best_count = len(result)
                best_result = result

    # Phase 3: Local search on best result
    if best_result is not None and time.time() < deadline:
        improved = _local_search(best_result, adj, all_edges_set, n, deadline)
        if improved and _validate(improved, adj, all_edges_set):
            if len(improved) < best_count:
                best_count = len(improved)
                best_result = improved

    # Phase 4: More local search iterations with different starting points
    for seed in range(10):
        if time.time() > deadline:
            break
        result = _greedy_partition(adj, all_edges_set, n, deadline, seed=100 + seed)
        if result and _validate(result, adj, all_edges_set):
            improved = _local_search(result, adj, all_edges_set, n, deadline, max_rounds=100)
            if improved and _validate(improved, adj, all_edges_set):
                if len(improved) < best_count:
                    best_count = len(improved)
                    best_result = improved

    # Final fallback
    if best_result is None:
        best_result = [({u}, {v}) for u, v in P['edges']]

    # Convert to proper types: list of [list(left), list(right)]
    result = []
    for left, right in best_result:
        result.append([list(left), list(right)])

    return result
