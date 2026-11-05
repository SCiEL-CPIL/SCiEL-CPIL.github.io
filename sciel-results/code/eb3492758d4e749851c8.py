import time
import random
from collections import defaultdict


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0

        edge_set = set(edges)
        used_edges = set()

        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)

        return used_edges == edge_set

    def greedy_target(edges, k, seed_val):
        """
        Attempts to cover all edges with exactly k simple paths.
        Returns the list of paths if successful, else None.
        """
        if k <= 0 and edges:
            return None
        if len(edges) < k:
            return None

        rng = random.Random(seed_val)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        paths = [[] for _ in range(k)]

        # Pick k distinct edges to start k paths
        edge_list = list(remaining)
        rng.shuffle(edge_list)

        start_edges = edge_list[:k]

        for i in range(k):
            u, v = start_edges[i]
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))
            paths[i] = [u, v]

        # Extend paths greedily
        changed = True
        while changed and remaining:
            changed = False
            idxs = list(range(k))
            rng.shuffle(idxs)

            for i in idxs:
                if not paths[i]:
                    continue

                # Try extending back
                back = paths[i][-1]
                candidates = []
                for nb in adj[back]:
                    if nb in paths[i]:
                        continue
                    if (back, nb) in remaining or (nb, back) in remaining:
                        candidates.append(nb)

                if candidates:
                    nb = rng.choice(candidates)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    paths[i].append(nb)
                    changed = True
                    continue

                # Try extending front
                front = paths[i][0]
                candidates = []
                for nb in adj[front]:
                    if nb in paths[i]:
                        continue
                    if (front, nb) in remaining or (nb, front) in remaining:
                        candidates.append(nb)

                if candidates:
                    nb = rng.choice(candidates)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    paths[i].insert(0, nb)
                    changed = True
                    continue

        if remaining:
            return None
        return paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency and compute lower bound
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        deg = {v: len(neigh) for v, neigh in adj.items()}
        odd_count = sum(1 for d in deg.values() if d % 2 == 1)
        lower_bound = max(1, (odd_count + 1) // 2)

        # Allocate time per instance
        time_per_instance = 42.0
        inst_start = time.time()
        inst_deadline = inst_start + time_per_instance

        best_paths = None
        best_count = float('inf')

        # Try targeting the lower bound first, then increase
        for target_k in range(lower_bound, min(len(edges), lower_bound + 5) + 1):
            if time.time() > inst_deadline - 2:
                break

            # Multiple restarts for this target
            max_restarts = 3000
            for restart in range(max_restarts):
                if time.time() > inst_deadline - 2:
                    break

                seed_val = seed * 100000 + restart
                paths = greedy_target(edges, target_k, seed_val)
                if paths is not None:
                    if validate_paths(paths, edges):
                        if target_k < best_count:
                            best_count = target_k
                            best_paths = paths
                        # Found optimal for this target, try next target
                        break

        if best_paths is None:
            # Fallback: greedy without target
            rng = random.Random(seed + 9999)
            remaining = set(edges)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)

            paths = []
            while remaining:
                degs = {}
                for u, v in remaining:
                    degs[u] = degs.get(u, 0) + 1
                    degs[v] = degs.get(v, 0) + 1
                if not degs:
                    break
                max_d = max(degs.values())
                cands = [v for v, d in degs.items() if d == max_d]
                start_u = rng.choice(cands)

                neighbors = [nb for nb in adj[start_u]
                             if (start_u, nb) in remaining or (nb, start_u) in remaining]
                if not neighbors:
                    e = next(iter(remaining))
                    start_u, start_v = e
                else:
                    start_v = rng.choice(neighbors)

                path = [start_u, start_v]
                if (start_u, start_v) in remaining:
                    remaining.remove((start_u, start_v))
                else:
                    remaining.remove((start_v, start_u))

                changed = True
                while changed:
                    changed = False
                    back = path[-1]
                    cands = [nb for nb in adj[back]
                             if nb not in path and
                             ((back, nb) in remaining or (nb, back) in remaining)]
                    if cands:
                        nb = rng.choice(cands)
                        if (back, nb) in remaining:
                            remaining.remove((back, nb))
                        else:
                            remaining.remove((nb, back))
                        path.append(nb)
                        changed = True
                        continue
                    front = path[0]
                    cands = [nb for nb in adj[front]
                             if nb not in path and
                             ((front, nb) in remaining or (nb, front) in remaining)]
                    if cands:
                        nb = rng.choice(cands)
                        if (front, nb) in remaining:
                            remaining.remove((front, nb))
                        else:
                            remaining.remove((nb, front))
                        path.insert(0, nb)
                        changed = True
                        continue

                paths.append(path)

            return paths

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []

    for idx, spec in enumerate(specs):
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        paths = solve_instance(n, p, seed)
        results.append(paths)

    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results


if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
