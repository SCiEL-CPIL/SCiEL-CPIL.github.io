"""Path decomposition via greedy construction + matching-based path merging."""
import time
from collections import defaultdict
from random import Random
from itertools import combinations

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _greedy_initial_paths(edge_set, adj, rng, deadline):
    """Build initial set of simple paths via greedy maximal path construction."""
    remaining = set(edge_set)
    paths = []

    rem_adj = defaultdict(list)
    for u, v in remaining:
        rem_adj[u].append(v)
        rem_adj[v].append(u)

    def rem_degree(v):
        return len(rem_adj[v]) if v in rem_adj else 0

    while remaining and time.time() < deadline:
        rem_odd = [u for u in rem_adj if rem_degree(u) % 2 == 1]
        if rem_odd:
            candidates = sorted(rem_odd, key=lambda u: -rem_degree(u))[:4]
            start = rng.choice(candidates)
        else:
            candidates = [u for u in rem_adj if rem_degree(u) > 0]
            if not candidates:
                break
            candidates = sorted(candidates, key=lambda u: -rem_degree(u))[:4]
            start = rng.choice(candidates)

        path = [start]
        in_path = {start}
        current = start

        while True:
            neighbors = [nb for nb in rem_adj[current]
                         if (min(current, nb), max(current, nb)) in remaining]
            if not neighbors:
                break

            non_cycle = [nb for nb in neighbors if nb not in in_path]
            if non_cycle:
                non_cycle.sort(key=lambda v: -rem_degree(v))
                if rng.random() < 0.25 and len(non_cycle) > 1:
                    next_v = non_cycle[rng.randrange(1, min(3, len(non_cycle)))]
                else:
                    next_v = non_cycle[0]
            else:
                break

            key = (min(current, next_v), max(current, next_v))
            remaining.discard(key)
            path.append(next_v)
            in_path.add(next_v)

            if next_v in rem_adj[current]:
                rem_adj[current].remove(next_v)
            if current in rem_adj[next_v]:
                rem_adj[next_v].remove(current)
            if not rem_adj[current]:
                del rem_adj[current]
            if not rem_adj[next_v]:
                del rem_adj[next_v]

            current = next_v

        if len(path) > 1:
            paths.append(path)

    while remaining and time.time() < deadline:
        e = next(iter(remaining))
        u, v = e
        paths.append([u, v])
        remaining.discard(e)
        if v in rem_adj.get(u, []):
            rem_adj[u].remove(v)
        if u in rem_adj.get(v, []):
            rem_adj[v].remove(u)
        if u in rem_adj and not rem_adj[u]:
            del rem_adj[u]
        if v in rem_adj and not rem_adj[v]:
            del rem_adj[v]

    return paths


def _merge_paths_via_matching(paths, edge_set, adj, rng, deadline):
    """
    Iteratively merge paths using maximum matching on path endpoints.
    Two paths can be merged if:
      - They share no vertices (to keep the result simple)
      - There exists an edge in the graph connecting an endpoint of path A
        to an endpoint of path B (specifically: tail of one to head of other,
        or we reverse one path first)
    We build a compatibility graph on path endpoints and find a matching.
    """
    if len(paths) <= 1:
        return paths

    # Precompute vertex sets for each path
    path_verts = [set(p) for p in paths]

    # For each pair of paths, check if they can be merged.
    # Paths i and j can be merged if:
    #   - path_verts[i] & path_verts[j] == empty
    #   - There's an edge between {paths[i][0], paths[i][-1]} and {paths[j][0], paths[j][-1]}
    #     specifically connecting an endpoint of i to an endpoint of j.
    # The merge produces a new simple path (concatenation, possibly reversing one).

    # Build adjacency of endpoints: for each path, its two endpoints.
    # We'll build a graph where nodes are (path_index, endpoint_side) where side=0 is head, side=1 is tail.
    # An edge exists between (i, s1) and (j, s2) if paths i and j are vertex-disjoint
    # and there's a graph edge between paths[i][s1] and paths[j][s2].
    # But we need to be careful: merging means concatenating. If we connect tail of i to head of j,
    # we get path i + path j. If we connect head of i to head of j, we reverse one.

    # Simpler: for each pair (i, j), check all 4 endpoint combinations.
    # If any combination works (edge exists and vertex-disjoint), mark them as mergeable.
    # For matching, we want to select a set of pairs to merge simultaneously.
    # But merging changes the paths, so we do it iteratively: find a matching, merge all matched pairs, repeat.

    # To make this tractable, at each iteration:
    # 1. Build a graph on path indices where edge (i,j) exists if paths can be merged.
    # 2. Find a maximum matching in this graph.
    # 3. Merge all matched pairs.
    # 4. Repeat until no more merges possible or deadline.

    def can_merge(i, j):
        """Check if paths i and j can be merged into a single simple path."""
        if path_verts[i] & path_verts[j]:
            return None
        # Check 4 endpoint connections
        ei_head, ei_tail = paths[i][0], paths[i][-1]
        ej_head, ej_tail = paths[j][0], paths[j][-1]
        # Connection: tail of i to head of j -> i + j
        if (min(ei_tail, ej_head), max(ei_tail, ej_head)) in edge_set:
            return (i, j, 0, 0)  # tail_i to head_j, no reversal
        # head of i to head of j -> reverse i, then + j
        if (min(ei_head, ej_head), max(ei_head, ej_head)) in edge_set:
            return (i, j, 1, 0)  # head_i to head_j, reverse i
        # tail of i to tail of j -> i + reverse j
        if (min(ei_tail, ej_tail), max(ei_tail, ej_tail)) in edge_set:
            return (i, j, 0, 1)  # tail_i to tail_j, reverse j
        # head of i to tail of j -> reverse i + j ... wait
        if (min(ei_head, ej_tail), max(ei_head, ej_tail)) in edge_set:
            return (i, j, 1, 1)  # head_i to tail_j, reverse i
        return None

    current_paths = [list(p) for p in paths]
    current_verts = [set(p) for p in current_paths]

    for iteration in range(50):  # Cap iterations
        if time.time() > deadline:
            break
        if len(current_paths) <= 1:
            break

        # Build compatibility graph
        n_paths = len(current_paths)
        compat = defaultdict(set)
        merge_info = {}

        for i in range(n_paths):
            for j in range(i + 1, n_paths):
                info = can_merge(i, j)
                if info is not None:
                    compat[i].add(j)
                    compat[j].add(i)
                    merge_info[(i, j)] = info

        if not any(compat.values()):
            break

        # Find maximum matching using a simple greedy + local search
        # For small graphs, use a simple approach: try all matchings via backtracking
        # But that's exponential. Instead, use a greedy matching with randomization.

        # Simple greedy matching:
        matched = set()
        matching = []
        indices = list(range(n_paths))
        rng.shuffle(indices)

        for i in indices:
            if i in matched:
                continue
            candidates = [j for j in compat.get(i, []) if j not in matched]
            if candidates:
                # Prefer the one with fewest other options (to preserve flexibility)
                candidates.sort(key=lambda j: len([k for k in compat.get(j, []) if k not in matched]))
                j = candidates[0]
                matching.append((i, j))
                matched.add(i)
                matched.add(j)

        if not matching:
            break

        # Apply merges
        new_paths = []
        used = set()
        for i, j in matching:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            info = merge_info.get((i, j)) or merge_info.get((j, i))
            if info is None:
                # Re-check
                info = can_merge(i, j)
                if info is None:
                    new_paths.append(current_paths[i])
                    new_paths.append(current_paths[j])
                    continue

            pi, pj = current_paths[i], current_paths[j]
            # info = (i, j, rev_i, rev_j) but i,j here are the original indices in the call
            # Actually info was computed with (i, j) as passed to can_merge
            # Let me re-derive: the 4th and 5th elements indicate which endpoint of i and j connect
            # 0 = tail, 1 = head
            # If rev_i=0: use pi as-is; rev_i=1: reverse pi
            # If rev_j=0: use pj as-is; rev_j=1: reverse pj
            # The connection is between the specified endpoints.

            # Let me just redo the logic cleanly:
            # We know there's an edge between some endpoint of pi and some endpoint of pj.
            # Let's find which.
            ei_head, ei_tail = pi[0], pi[-1]
            ej_head, ej_tail = pj[0], pj[-1]

            merged = None
            # tail_i to head_j
            if (min(ei_tail, ej_head), max(ei_tail, ej_head)) in edge_set:
                merged = pi + pj
            # head_i to head_j (reverse i)
            elif (min(ei_head, ej_head), max(ei_head, ej_head)) in edge_set:
                merged = pi[::-1] + pj
            # tail_i to tail_j (reverse j)
            elif (min(ei_tail, ej_tail), max(ei_tail, ej_tail)) in edge_set:
                merged = pi + pj[::-1]
            # head_i to tail_j (reverse i)
            elif (min(ei_head, ej_tail), max(ei_head, ej_tail)) in edge_set:
                merged = pi[::-1] + pj

            if merged is not None:
                # Verify simplicity
                if len(merged) == len(set(merged)):
                    new_paths.append(merged)
                else:
                    new_paths.append(pi)
                    new_paths.append(pj)
            else:
                new_paths.append(pi)
                new_paths.append(pj)

        for i in range(len(current_paths)):
            if i not in used:
                new_paths.append(current_paths[i])

        current_paths = new_paths
        current_verts = [set(p) for p in current_paths]

    return current_paths


def decompose_paths(edges, deadline):
    if not edges:
        return []

    edge_set = set()
    adj = defaultdict(list)
    for e in edges:
        u, v = e
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)

    degree = {u: len(adj[u]) for u in adj}
    odd_vertices = [u for u in degree if degree[u] % 2 == 1]
    min_paths = max(1, len(odd_vertices) // 2)

    best_paths = None
    best_count = float('inf')

    max_restarts = 40

    for trial in range(max_restarts):
        if time.time() > deadline:
            break

        rng = Random(trial * 7919 + 13)
        initial = _greedy_initial_paths(edge_set, adj, rng, deadline)

        # Try to improve via matching-based merging
        merged = _merge_paths_via_matching(initial, edge_set, adj, rng, deadline)

        count = len(merged)
        if count < best_count:
            best_count = count
            best_paths = merged

        if best_count == min_paths:
            break

    if best_paths is None:
        best_paths = [[u, v] for u, v in edge_set]

    return best_paths


def construct():
    deadline = time.time() + 110

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue

        edges = instance(*spec)
        paths = decompose_paths(edges, deadline)
        results.append(paths)

    return results
