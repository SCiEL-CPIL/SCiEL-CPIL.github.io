"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']

def _build_adj():
    adj = [set() for _ in range(N)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    return adj, edge_set

def _cycle_edges(cycle):
    edges = set()
    m = len(cycle)
    for i in range(m):
        a = cycle[i]
        b = cycle[(i + 1) % m]
        edges.add((min(a, b), max(a, b)))
    return edges

def _is_hamiltonian_cycle(cycle, adj, edge_set):
    if len(cycle) != N:
        return False
    if len(set(cycle)) != N:
        return False
    for v in cycle:
        if v < 0 or v >= N:
            return False
    m = len(cycle)
    for i in range(m):
        a = cycle[i]
        b = cycle[(i + 1) % m]
        if (min(a, b), max(a, b)) not in edge_set:
            return False
    return True

def _find_hamiltonian(adj, used_edges, start=0, deadline=None):
    # Backtracking Hamiltonian cycle search with pruning
    path = [start]
    visited = {start}
    # used_edges is a set of (u,v) tuples already consumed

    def dfs():
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == N:
            last = path[-1]
            if (min(last, start), max(last, start)) in used_edges:
                return None
            return list(path)
        cur = path[-1]
        # candidates: neighbors not visited, edge not used
        cands = []
        for nb in adj[cur]:
            if nb in visited:
                continue
            e = (min(cur, nb), max(cur, nb))
            if e in used_edges:
                continue
            cands.append(nb)
        # heuristic ordering: prefer vertices with fewer available exits
        def score(nb):
            avail = 0
            for nn in adj[nb]:
                if nn in visited:
                    continue
                e = (min(nb, nn), max(nb, nn))
                if e in used_edges:
                    continue
                avail += 1
            return avail
        cands.sort(key=score)
        for nb in cands:
            e = (min(cur, nb), max(cur, nb))
            path.append(nb)
            visited.add(nb)
            used_edges.add(e)
            res = dfs()
            if res is not None:
                return res
            used_edges.discard(e)
            visited.discard(nb)
            path.pop()
        return None

    return dfs()

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin under 90s
    adj, edge_set = _build_adj()

    # Try to find 3 edge-disjoint Hamiltonian cycles
    # Strategy: greedy with backtracking on the choice of first cycle
    best_cycles = []

    # Precompute: try a few different starts / orderings
    attempts = 0
    found = []
    used_edges = set()

    # First cycle: try to find one
    # We'll try multiple Hamiltonian cycles and pick a set of 3 that are edge-disjoint
    # Approach: enumerate candidate cycles (limited), then pick max edge-disjoint subset

    candidate_pool = []
    pool_deadline = start_time + 50.0

    # Generate candidate Hamiltonian cycles using randomized restarts
    rng = random.Random(42)
    # Try different start vertices and neighbor orderings
    for trial in range(200):
        if time.time() > pool_deadline:
            break
        start = rng.randint(0, N - 1)
        # Shuffle adj lists temporarily for diversity
        shuffled_adj = []
        for v in range(N):
            lst = list(adj[v])
            rng.shuffle(lst)
            shuffled_adj.append(lst)
        # Run DFS with shuffled adjacency
        # Inline DFS with shuffled adj
        path = [start]
        visited = {start}
        used = set()
        found_cycle = [False]

        def dfs2():
            if time.time() > pool_deadline:
                return
            if len(path) == N:
                last = path[-1]
                e = (min(last, start), max(last, start))
                if e in used:
                    return
                found_cycle[0] = True
                return
            cur = path[-1]
            cands = []
            for nb in shuffled_adj[cur]:
                if nb in visited:
                    continue
                e = (min(cur, nb), max(cur, nb))
                if e in used:
                    continue
                cands.append(nb)
            # sort by fewest exits
            def score(nb):
                avail = 0
                for nn in shuffled_adj[nb]:
                    if nn in visited:
                        continue
                    e = (min(nb, nn), max(nb, nn))
                    if e in used:
                        continue
                    avail += 1
                return avail
            cands.sort(key=score)
            for nb in cands:
                if found_cycle[0]:
                    return
                e = (min(cur, nb), max(cur, nb))
                path.append(nb)
                visited.add(nb)
                used.add(e)
                dfs2()
                if found_cycle[0]:
                    return
                used.discard(e)
                visited.discard(nb)
                path.pop()

        dfs2()
        if found_cycle[0]:
            cyc = list(path)
            # normalize: rotate so min vertex first, and fix direction
            min_idx = cyc.index(min(cyc))
            cyc = cyc[min_idx:] + cyc[:min_idx]
            # canonical direction: make second vertex smaller than last
            rev = list(reversed(cyc))
            if rev[1] < rev[-1]:
                cyc = rev
            key = tuple(cyc)
            if key not in candidate_pool:
                candidate_pool.append(key)
                if len(candidate_pool) >= 30:
                    break

    # Now select up to 3 edge-disjoint cycles from the pool
    # Greedy: pick cycles that don't share edges
    selected = []
    used_edges = set()
    # Sort candidates by some heuristic (e.g., by edge set size - all same)
    # Try all permutations of small pool
    pool = candidate_pool
    # Greedy with backtracking over pool
    def select(pool, used, selected, depth, max_depth):
        if len(selected) == max_depth:
            return list(selected)
        if not pool:
            return list(selected) if selected else None
        best = None
        for i, cyc in enumerate(pool):
            ce = _cycle_edges(list(cyc))
            if ce & used:
                continue
            new_used = used | ce
            new_selected = selected + [list(cyc)]
            rest = pool[:i] + pool[i+1:]
            res = select(rest, new_used, new_selected, depth + 1, max_depth)
            if res is not None and (best is None or len(res) > len(best)):
                best = res
            if best is not None and len(best) == max_depth:
                return best
        return best if best is not None else (list(selected) if selected else None)

    result = select(pool, set(), [], 0, 3)
    if result:
        best_cycles = result

    # If we only got 1 or 2, try to extend with fresh searches
    if len(best_cycles) < 3:
        used_edges = set()
        for c in best_cycles:
            used_edges |= _cycle_edges(c)
        # Try to find more cycles avoiding used edges
        for extra in range(3 - len(best_cycles)):
            if time.time() > deadline - 5.0:
                break
            found_extra = None
            # randomized search for a Hamiltonian cycle avoiding used_edges
            for trial in range(50):
                if time.time() > deadline - 5.0:
                    break
                start = random.randint(0, N - 1)
                shuffled_adj = []
                for v in range(N):
                    lst = list(adj[v])
                    random.shuffle(lst)
                    shuffled_adj.append(lst)
                path = [start]
                visited = {start}
                used = set(used_edges)
                found_cycle = [False]

                def dfs3():
                    if time.time() > deadline - 5.0:
                        return
                    if len(path) == N:
                        last = path[-1]
                        e = (min(last, start), max(last, start))
                        if e in used:
                            return
                        found_cycle[0] = True
                        return
                    cur = path[-1]
                    cands = []
                    for nb in shuffled_adj[cur]:
                        if nb in visited:
                            continue
                        e = (min(cur, nb), max(cur, nb))
                        if e in used:
                            continue
                        cands.append(nb)
                    def score(nb):
                        avail = 0
                        for nn in shuffled_adj[nb]:
                            if nn in visited:
                                continue
                            e = (min(nb, nn), max(nb, nn))
                            if e in used:
                                continue
                            avail += 1
                        return avail
                    cands.sort(key=score)
                    for nb in cands:
                        if found_cycle[0]:
                            return
                        e = (min(cur, nb), max(cur, nb))
                        path.append(nb)
                        visited.add(nb)
                        used.add(e)
                        dfs3()
                        if found_cycle[0]:
                            return
                        used.discard(e)
                        visited.discard(nb)
                        path.pop()

                dfs3()
                if found_cycle[0]:
                    cyc = list(path)
                    min_idx = cyc.index(min(cyc))
                    cyc = cyc[min_idx:] + cyc[:min_idx]
                    rev = list(reversed(cyc))
                    if rev[1] < rev[-1]:
                        cyc = rev
                    # check edge-disjoint from best_cycles
                    ce = _cycle_edges(cyc)
                    if not (ce & used_edges):
                        found_extra = cyc
                        break
            if found_extra:
                best_cycles.append(found_extra)
                used_edges |= _cycle_edges(found_extra)
            else:
                break

    # Fallback: if we have nothing, return a single valid cycle
    if not best_cycles:
        # Find any Hamiltonian cycle
        cyc = _find_hamiltonian(adj, set(), start=0, deadline=deadline)
        if cyc is None:
            # Should not happen; return a trivial valid cycle if possible
            # Actually we must return nonempty list of valid cycles
            # Try a simple cycle if Hamiltonian fails
            # But graph is connected, Hamiltonian should exist
            return []
        best_cycles = [cyc]

    # Validate and convert to proper types
    final = []
    for c in best_cycles:
        c = [int(v) for v in c]
        final.append(c)

    # Final validation
    seen_edges = set()
    valid_final = []
    for c in final:
        if not _is_hamiltonian_cycle(c, adj, edge_set):
            continue
        ce = _cycle_edges(c)
        if ce & seen_edges:
            continue
        seen_edges |= ce
        valid_final.append(c)
        if len(valid_final) >= 10:
            break

    if not valid_final:
        # last resort
        cyc = _find_hamiltonian(adj, set(), start=0, deadline=deadline)
        if cyc is not None:
            valid_final = [[int(v) for v in cyc]]

    return valid_final
# EVOLVE-BLOCK-END
