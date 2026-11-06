import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 138.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def decompose_graph(edges, time_limit):
        if not edges:
            return []

        remaining_edges = set(edges)
        pieces = []

        def rebuild_adj():
            adj = defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            return adj

        def remove_cycle_edges(cycle):
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i + 1]
                if (u, v) in remaining_edges:
                    remaining_edges.remove((u, v))
                elif (v, u) in remaining_edges:
                    remaining_edges.remove((v, u))
            u, v = cycle[-1], cycle[0]
            if (u, v) in remaining_edges:
                remaining_edges.remove((u, v))
            elif (v, u) in remaining_edges:
                remaining_edges.remove((v, u))

        def find_longest_cycle_greedy(adj, seed, dl):
            """Find a long cycle using greedy longest-path extension."""
            if not remaining_edges:
                return None

            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                return None
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)

            best_cycle = None
            best_len = 0
            rng = random.Random(seed)

            for start in candidates[:12]:
                if time.time() > dl:
                    break
                for attempt in range(3):
                    if time.time() > dl:
                        break
                    visited = {start}
                    path = [start]
                    current = start

                    while True:
                        neighbors = [n for n in adj[current] if n not in visited]
                        if not neighbors:
                            break
                        neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                        top_k = min(3, len(neighbors))
                        nxt = rng.choice(neighbors[:top_k])
                        visited.add(nxt)
                        path.append(nxt)
                        current = nxt

                    # Check if cycle can be closed
                    if len(path) >= 3 and path[-1] in adj[path[0]]:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)

                    # Look for any back-edge to form a cycle
                    for i in range(len(path) - 2):
                        if time.time() > dl:
                            break
                        for j in range(len(path) - 1, i + 1, -1):
                            if path[i] in adj[path[j]]:
                                cycle = path[i:j + 1]
                                if len(cycle) >= 3 and len(cycle) > best_len:
                                    best_len = len(cycle)
                                    best_cycle = list(cycle)
                                break

            return best_cycle

        def find_cycle_2opt(adj, seed, dl):
            """Try to find a cycle using a 2-opt style improvement."""
            if not remaining_edges:
                return None

            vertices = [v for v in adj if adj[v]]
            if len(vertices) < 3:
                return None

            rng = random.Random(seed)

            # Try to find a triangle first
            for v in vertices:
                if time.time() > dl:
                    break
                for u in adj[v]:
                    if u <= v:
                        continue
                    for w in adj[u]:
                        if w <= u:
                            continue
                        if w in adj[v]:
                            return [v, u, w]

            # Try to build a cycle by extending a path
            best_cycle = None
            best_len = 0
            for _ in range(20):
                if time.time() > dl:
                    break
                start = rng.choice(vertices)
                path = [start]
                visited = {start}

                for _ in range(50):
                    u = path[-1]
                    candidates = [v for v in adj[u] if v not in visited]
                    if not candidates:
                        break
                    scored = []
                    for v in candidates:
                        score = len(adj[v])
                        if v in adj[path[0]]:
                            score += 100
                        scored.append((score, v))
                    scored.sort(reverse=True)
                    _, next_v = scored[0]
                    path.append(next_v)
                    visited.add(next_v)

                # Check for cycle
                if len(path) >= 3 and path[0] in adj[path[-1]]:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = list(path)
                else:
                    # Check for any cycle in path
                    for i in range(len(path) - 2):
                        for j in range(len(path) - 1, i, -1):
                            if path[i] in adj[path[j]]:
                                cycle = path[i:j + 1]
                                if len(cycle) >= 3 and len(cycle) > best_len:
                                    best_len = len(cycle)
                                    best_cycle = list(cycle)
                                break

            return best_cycle

        def find_triangles(adj):
            """Find all triangles in the remaining graph."""
            triangles = []
            for u in adj:
                for v in adj[u]:
                    if v <= u:
                        continue
                    for w in adj[v]:
                        if w <= v:
                            continue
                        if w in adj[u]:
                            triangles.append([u, v, w])
            return triangles

        # Main decomposition loop
        max_iterations = 500
        iteration = 0
        while remaining_edges and iteration < max_iterations:
            if time.time() > time_limit:
                break
            iteration += 1

            adj = rebuild_adj()

            # Try multiple cycle-finding strategies and take the best
            c1 = find_longest_cycle_greedy(adj, iteration * 7, time_limit)
            c2 = find_cycle_2opt(adj, iteration * 13, time_limit)

            if c1 and c2:
                cycle = c1 if len(c1) >= len(c2) else c2
            elif c1:
                cycle = c1
            elif c2:
                cycle = c2
            else:
                cycle = None

            if cycle:
                pieces.append(cycle)
                remove_cycle_edges(cycle)
            else:
                # Try to find triangles
                triangles = find_triangles(adj)
                if triangles:
                    best_tri = triangles[0]
                    best_score = sum(len(adj[v]) for v in best_tri)
                    for tri in triangles:
                        score = sum(len(adj[v]) for v in tri)
                        if score > best_score:
                            best_score = score
                            best_tri = tri
                    pieces.append(best_tri)
                    remove_cycle_edges(best_tri)
                else:
                    # No cycles left; add remaining edges individually
                    for e in list(remaining_edges):
                        pieces.append(list(e))
                        remaining_edges.remove(e)
                    break

        # If we still have edges (time ran out), add them individually
        for e in list(remaining_edges):
            pieces.append(list(e))
            remaining_edges.remove(e)

        return pieces

    results = []
    time_limits = [35.0, 40.0, 45.0]

    for i, edges in enumerate(graphs):
        global_remaining = deadline - time.time()
        if global_remaining < 2.0:
            for j in range(i, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break

        current_limit = min(time_limits[i], global_remaining - 2.0)
        if current_limit < 1.0:
            current_limit = 1.0

        time_limit = time.time() + current_limit
        pieces = decompose_graph(edges, time_limit)
        results.append(pieces)

    return results


if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)

        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k + 1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break

        if valid and used_edges != edge_set:
            valid = False

        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
    total = sum(len(r) for r in res)
    print(f"Total pieces: {total}")
