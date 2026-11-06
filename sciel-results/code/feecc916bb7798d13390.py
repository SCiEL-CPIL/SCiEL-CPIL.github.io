import time
import random
from collections import defaultdict


def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    start_time = time.time()
    deadline = start_time + 110

    def time_up():
        return time.time() > deadline

    for n, p, seed in specs:
        edges = instance(n, p, seed)
        remaining = set(edges)
        pieces = []

        def build_adj(edge_set):
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        adj = build_adj(remaining)

        def find_longest_cycle(adj, remaining):
            if not remaining:
                return None

            active_vertices = set()
            for u, v in remaining:
                active_vertices.add(u)
                active_vertices.add(v)

            if len(active_vertices) < 3:
                return None

            best_cycle = None
            best_len = 0

            num_restarts = 100
            for r in range(num_restarts):
                if time_up():
                    break

                verts = list(active_vertices)
                random.shuffle(verts)
                start = verts[0]

                path = [start]
                visited = {start}
                current = start

                if r % 3 == 0:
                    strategy = 'max_deg'
                elif r % 3 == 1:
                    strategy = 'min_deg'
                else:
                    strategy = 'random'

                while True:
                    if time_up():
                        break

                    neighbors = [v for v in adj[current]
                                 if (min(current, v), max(current, v)) in remaining]
                    if not neighbors:
                        break

                    unvisited = [v for v in neighbors if v not in visited]

                    if unvisited:
                        if strategy == 'max_deg':
                            next_node = max(unvisited, key=lambda x: len(
                                [y for y in adj[x] if (min(x, y), max(x, y)) in remaining]))
                        elif strategy == 'min_deg':
                            next_node = min(unvisited, key=lambda x: len(
                                [y for y in adj[x] if (min(x, y), max(x, y)) in remaining]))
                        else:
                            next_node = random.choice(unvisited)
                        path.append(next_node)
                        visited.add(next_node)
                        current = next_node
                    else:
                        if len(path) >= 3 and (min(current, path[0]), max(current, path[0])) in remaining:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = path[:]
                        break

            return best_cycle

        while remaining and not time_up():
            cycle = find_longest_cycle(adj, remaining)

            if cycle and len(cycle) >= 3:
                cycle_edges = set()
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        cycle_edges.add(e)
                    else:
                        valid = False
                        break

                if valid:
                    pieces.append(cycle)
                    remaining -= cycle_edges
                    adj = build_adj(remaining)
                else:
                    for e in list(remaining):
                        pieces.append(list(e))
                    remaining = set()
                    break
            else:
                for e in remaining:
                    pieces.append(list(e))
                remaining = set()
                break

        results.append(pieces)

    return results
