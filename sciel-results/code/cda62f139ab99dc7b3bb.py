"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    def decompose_graph(n, edges):
        """
        Deterministic longest-path greedy with degree heuristics.
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining = set(edges)
        paths = []

        def path_edges(path):
            return set((min(path[i], path[i+1]), max(path[i], path[i+1])) for i in range(len(path)-1))

        def extract_greedy(start, rem):
            """Deterministic greedy: always pick the neighbor with highest remaining degree."""
            path = [start]
            visited = {start}
            cur = start
            while True:
                # Find candidates via unused edges
                cands = []
                for nb in adj[cur]:
                    e = (min(cur, nb), max(cur, nb))
                    if e in rem:
                        cands.append(nb)
                if not cands:
                    break
                # Prefer unvisited
                unvisited = [nb for nb in cands if nb not in visited]
                if unvisited:
                    # Pick the one with highest remaining degree
                    best_nb = max(unvisited, key=lambda x: len([e for e in rem if x in e]))
                else:
                    break
                e = (min(cur, best_nb), max(cur, best_nb))
                rem.discard(e)
                path.append(best_nb)
                visited.add(best_nb)
                cur = best_nb
            return path

        def extract_greedy_rev(start, rem):
            """Greedy preferring lower-degree neighbors to leave high-degree nodes for later."""
            path = [start]
            visited = {start}
            cur = start
            while True:
                cands = []
                for nb in adj[cur]:
                    e = (min(cur, nb), max(cur, nb))
                    if e in rem:
                        cands.append(nb)
                if not cands:
                    break
                unvisited = [nb for nb in cands if nb not in visited]
                if unvisited:
                    best_nb = min(unvisited, key=lambda x: len([e for e in rem if x in e]))
                else:
                    break
                e = (min(cur, best_nb), max(cur, best_nb))
                rem.discard(e)
                path.append(best_nb)
                visited.add(best_nb)
                cur = best_nb
            return path

        # Repeatedly extract longest path
        while remaining:
            # Try multiple strategies and pick the longest valid path
            best_path = None
            best_edges = None

            # Strategy 1: start from each endpoint of a random-ish edge, use high-degree greedy
            # Pick edge with max sum of degrees
            max_deg_sum = -1
            pick_edge = None
            for e in remaining:
                ds = len([x for x in remaining if e[0] in x]) + len([x for x in remaining if e[1] in x])
                if ds > max_deg_sum:
                    max_deg_sum = ds
                    pick_edge = e

            u, v = pick_edge
            # Try both directions with both heuristics
            for start, func in [(u, extract_greedy), (v, extract_greedy),
                                 (u, extract_greedy_rev), (v, extract_greedy_rev)]:
                rem_copy = set(remaining)
                p = func(start, rem_copy)
                pe = path_edges(p)
                # Validate: all edges in p must be in remaining
                if pe.issubset(remaining) and len(p) > 1:
                    if best_path is None or len(p) > len(best_path):
                        best_path = p
                        best_edges = pe

            if best_path is None:
                # Fallback: single edge
                best_path = [u, v]
                best_edges = {pick_edge}

            remaining -= best_edges
            paths.append(best_path)

        # Post-processing: try to merge adjacent paths
        def try_merge(paths_list):
            """Try to merge paths that share a vertex at endpoints."""
            merged = True
            while merged:
                merged = False
                new_paths = list(paths_list)
                for i in range(len(new_paths)):
                    for j in range(i+1, len(new_paths)):
                        pi = new_paths[i]
                        pj = new_paths[j]
                        # Check if we can concatenate
                        # pi end connects to pj start
                        if pi[-1] == pj[0]:
                            candidate = pi + pj[1:]
                            ce = path_edges(candidate)
                            # Check no repeated edges
                            if len(ce) == len(candidate) - 1:
                                new_paths[i] = candidate
                                new_paths.pop(j)
                                merged = True
                                break
                        # pi end connects to pj end (reverse pj)
                        if pi[-1] == pj[-1]:
                            candidate = pi + list(reversed(pj))[:-1]
                            ce = path_edges(candidate)
                            if len(ce) == len(candidate) - 1:
                                new_paths[i] = candidate
                                new_paths.pop(j)
                                merged = True
                                break
                        # pj end connects to pi start
                        if pj[-1] == pi[0]:
                            candidate = list(reversed(pj)) + pi[1:]
                            ce = path_edges(candidate)
                            if len(ce) == len(candidate) - 1:
                                new_paths[j] = candidate
                                new_paths.pop(i)
                                merged = True
                                break
                        # pj start connects to pi end
                        if pj[0] == pi[-1]:
                            candidate = pi + pj[1:]
                            ce = path_edges(candidate)
                            if len(ce) == len(candidate) - 1:
                                new_paths[i] = candidate
                                new_paths.pop(j)
                                merged = True
                                break
                    if merged:
                        break
                paths_list = new_paths
            return paths_list

        paths = try_merge(paths)
        return paths

    results = []
    for spec in specs:
        if time.time() > DEADLINE - 5:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose_graph(n, edges)
        results.append(paths)

    return results
# EVOLVE-BLOCK-END
