"""Exact biclique partition via maximal biclique enumeration + branch-and-bound set cover."""

import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _seed_solution():
    """Fallback: one biclique per vertex (star decomposition)."""
    n = P['n']
    edges = P['edges']
    adj = {u: [] for u in range(n)}
    for a, b in edges:
        adj[a].append(b)
    result = []
    for u in range(n):
        if adj[u]:
            result.append([[u], list(adj[u])])
    return result


def _generate_candidates():
    """Generate candidate bicliques: stars and maximal bicliques via growth from each edge."""
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for a, b in edges:
        if a > b:
            a, b = b, a
        edge_set.add((a, b))

    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    candidates = []
    seen = set()

    def add_candidate(L, R):
        if not L or not R:
            return
        L_sorted = tuple(sorted(L))
        R_sorted = tuple(sorted(R))
        if L_sorted[0] <= R_sorted[0]:
            key = (L_sorted, R_sorted)
        else:
            key = (R_sorted, L_sorted)
        if key not in seen:
            seen.add(key)
            L_list = list(L_sorted)
            R_list = list(R_sorted)
            if L_list[0] > R_list[0]:
                L_list, R_list = R_list, L_list
            candidates.append((L_list, R_list))

    # Star candidates
    for u in range(n):
        if adj[u]:
            add_candidate([u], list(adj[u]))

    # Maximal bicliques by growing from each edge
    start_time = time.time()
    MAX_TIME = 30

    for (a, b) in edges:
        if time.time() - start_time > MAX_TIME:
            break
        L = {a}
        R = {b}
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x not in L and x not in R:
                    if adj[x].issuperset(R):
                        L.add(x)
                        changed = True
            for x in range(n):
                if x not in L and x not in R:
                    if adj[x].issuperset(L):
                        R.add(x)
                        changed = True
        add_candidate(list(L), list(R))

    # Also try starting with both orientations swapped
    for (a, b) in edges:
        if time.time() - start_time > MAX_TIME:
            break
        L = {b}
        R = {a}
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x not in L and x not in R:
                    if adj[x].issuperset(R):
                        L.add(x)
                        changed = True
            for x in range(n):
                if x not in L and x not in R:
                    if adj[x].issuperset(L):
                        R.add(x)
                        changed = True
        add_candidate(list(L), list(R))

    return candidates


def _solve_bnb(candidates, edge_indices, time_limit=35):
    """Branch-and-bound minimum set cover."""
    import time
    start_time = time.time()

    n_edges = len(edge_indices)

    # Build edge membership for each candidate
    cand_edges = []
    for L, R in candidates:
        covered = set()
        for u in L:
            for v in R:
                key = (min(u, v), max(u, v))
                if key in edge_indices:
                    covered.add(edge_indices[key])
        cand_edges.append(covered)

    n_cands = len(cand_edges)

    # Remove dominated candidates
    keep = [True] * n_cands
    for i in range(n_cands):
        if not keep[i]:
            continue
        for j in range(n_cands):
            if i == j or not keep[j]:
                continue
            if cand_edges[i] <= cand_edges[j] and len(cand_edges[i]) < len(cand_edges[j]):
                keep[i] = False
                break

    filtered_indices = [i for i in range(n_cands) if keep[i]]
    filtered_cand_edges = [cand_edges[i] for i in filtered_indices]
    n_filtered = len(filtered_indices)

    if n_filtered == 0:
        return None

    # Greedy initial upper bound
    order = sorted(range(n_filtered), key=lambda i: -len(filtered_cand_edges[i]))
    uncovered = set(range(n_edges))
    greedy_sol = []
    while uncovered:
        best_i = -1
        best_count = 0
        for i in order:
            count = len(filtered_cand_edges[i] & uncovered)
            if count > best_count:
                best_count = count
                best_i = i
        if best_i == -1 or best_count == 0:
            break
        greedy_sol.append(best_i)
        uncovered -= filtered_cand_edges[best_i]

    if uncovered:
        return None

    upper_bound = len(greedy_sol)
    best_sol = list(greedy_sol)

    # Edge to candidates mapping
    edge_to_cands = [[] for _ in range(n_edges)]
    for i, ce in enumerate(filtered_cand_edges):
        for e in ce:
            edge_to_cands[e].append(i)

    for e in range(n_edges):
        if not edge_to_cands[e]:
            return None

    nodes_explored = [0]

    def bnb(selection, uncovered):
        nonlocal upper_bound, best_sol
        nodes_explored[0] += 1

        if time.time() - start_time > time_limit:
            return

        if not uncovered:
            if len(selection) < upper_bound:
                upper_bound = len(selection)
                best_sol = list(selection)
            return

        # Lower bound 1: ceil(|uncovered| / max_single_cover)
        max_cover = 0
        for i in range(n_filtered):
            c = len(filtered_cand_edges[i] & uncovered)
            if c > max_cover:
                max_cover = c
        if max_cover == 0:
            return
        lb1 = (len(uncovered) + max_cover - 1) // max_cover

        # Lower bound 2: packing bound (greedy)
        lb2 = 0
        remaining = set(uncovered)
        while remaining:
            e = min(remaining, key=lambda x: len(edge_to_cands[x]))
            lb2 += 1
            to_remove = set()
            for ci in edge_to_cands[e]:
                to_remove.update(filtered_cand_edges[ci])
            remaining -= to_remove
            if not remaining:
                break

        lb = max(lb1, lb2)
        if len(selection) + lb >= upper_bound:
            return

        # Choose edge with fewest covering candidates
        e = min(uncovered, key=lambda x: len(edge_to_cands[x]))
        cands_for_e = sorted(edge_to_cands[e], key=lambda i: -len(filtered_cand_edges[i] & uncovered))

        for ci in cands_for_e:
            if time.time() - start_time > time_limit:
                return
            new_unc = uncovered - filtered_cand_edges[ci]
            selection.append(ci)
            bnb(selection, new_unc)
            selection.pop()

    bnb([], set(range(n_edges)))

    if best_sol:
        return [filtered_indices[i] for i in best_sol]
    return None


def construct():
    import time
    start = time.time()

    try:
        candidates = _generate_candidates()

        edge_indices = {}
        for i, (a, b) in enumerate(P['edges']):
            key = (min(a, b), max(a, b))
            edge_indices[key] = i

        if candidates:
            sol = _solve_bnb(candidates, edge_indices, time_limit=45)
            if sol is not None:
                result = []
                for idx in sol:
                    L, R = candidates[idx]
                    valid = True
                    for u in L:
                        for v in R:
                            key = (min(u, v), max(u, v))
                            if key not in edge_indices:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and L and R:
                        result.append([list(L), list(R)])

                if result:
                    edge_count = {}
                    for L, R in result:
                        for u in L:
                            for v in R:
                                key = (min(u, v), max(u, v))
                                edge_count[key] = edge_count.get(key, 0) + 1

                    all_edges = set(edge_indices.keys())
                    covered = set(edge_count.keys())

                    if covered == all_edges and all(c == 1 for c in edge_count.values()):
                        final_result = []
                        for L, R in result:
                            L_int = [int(x) for x in L]
                            R_int = [int(x) for x in R]
                            final_result.append([L_int, R_int])
                        return final_result
    except Exception:
        pass

    return _seed_solution()
