"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        edges_set = set(tuple(sorted(e)) for e in edges)
        n = max(max(e) for e in edges_set) + 1 if edges_set else 0

        def build_adj(rem):
            adj = {v: [] for v in range(n)}
            for (u, v) in rem:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        adj = build_adj(edges_set)
        pieces = []
        cycles_found = 0

        while edges_set and cycles_found < 200:
            # Find a cycle via DFS
            cycle = None
            for start in range(n):
                if not adj[start]:
                    continue
                # DFS
                stack = [(start, [start])]
                visited_in_path = {start}
                found = False
                while stack and not found:
                    node, path = stack.pop()
                    for nb in adj[node]:
                        edge = tuple(sorted((node, nb)))
                        if edge not in edges_set:
                            continue
                        if nb == start and len(path) >= 3:
                            cycle = list(path)
                            found = True
                            break
                        if nb not in visited_in_path:
                            visited_in_path.add(nb)
                            stack.append((nb, path + [nb]))
                if cycle:
                    break

            if cycle is None:
                break

            # Remove cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                edges_set.discard(e)
                adj[u].remove(v)
                adj[v].remove(u)

            pieces.append(list(cycle))
            cycles_found += 1

        # Remaining edges as single pieces
        for e in edges_set:
            pieces.append(list(e))

        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            results.append([list(e) for e in instance(*spec)])
            continue
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
