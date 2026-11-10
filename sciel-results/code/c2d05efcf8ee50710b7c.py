import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit

    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def get_cycle_edges(cycle):
        es = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            es.add((min(u, v), max(u, v)))
        return es

    def find_ham_cycle(used_edges, start=0, time_budget=None):
        if time_budget is None:
            time_budget = deadline - 5
        if time.time() > time_budget:
            return None

        # Build remaining adjacency
        rem_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    rem_adj[u].add(v)

        # Check min degree >= 2
        for u in range(n):
            if len(rem_adj[u]) < 2:
                return None

        # Check connectivity
        visited_conn = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited_conn:
                continue
            visited_conn.add(node)
            for nb in rem_adj[node]:
                if nb not in visited_conn:
                    stack.append(nb)
        if len(visited_conn) != n:
            return None

        path = [start]
        visited_set = {start}
        edges_used_local = set()

        # Precompute edge lookup for speed
        def edge_key(u, v):
            return (min(u, v), max(u, v))

        def dfs(current):
            if time.time() > time_budget:
                return None

            if len(path) == n:
                last = path[-1]
                e = edge_key(last, start)
                if e not in edges_used_local and e in edge_set:
                    return list(path)
                return None

            current = path[-1]
            # Sort neighbors by degree in remaining graph (MRV-like)
            candidates = []
            for nb in rem_adj[current]:
                if nb not in visited_set:
                    e = edge_key(current, nb)
                    if e not in edges_used_local and e in edge_set:
                        candidates.append((len(rem_adj[nb]), nb))

            candidates.sort()

            for _, nb in candidates:
                e = edge_key(current, nb)
                if e in edges_used_local:
                    continue

                # Pruning: check feasibility for unvisited vertices
                # Each unvisited vertex must still have at least 2 available edges
                # in the remaining graph (considering edges already used locally)
                feasible = True
                for v in range(n):
                    if v in visited_set or v == nb:
                        continue
                    # Count available edges from v to unvisited vertices or start
                    avail = 0
                    for nb2 in rem_adj[v]:
                        if nb2 in visited_set and nb2 != start:
                            # Edge to a visited non-start vertex: only usable if nb2 is current (the endpoint)
                            # Actually, in a Hamiltonian path, internal visited vertices can't be reconnected
                            # Only start and current (path endpoints) can accept new edges
                            continue
                        e2 = edge_key(v, nb2)
                        if e2 not in edges_used_local:
                            avail += 1
                    # v needs at least 2 connections in the final cycle
                    # But some of those may go to start or current (endpoints)
                    # Simplified check: avail >= 1 is necessary (very loose)
                    # Stronger: if v is not adjacent to start in remaining graph, it needs >= 2 among unvisited
                    if avail < 2:
                        # Check if v can connect to start
                        e_vs = edge_key(v, start)
                        if e_vs in edges_used_local or e_vs not in edge_set:
                            feasible = False
                            break
                        # v can use one edge to start, so needs >= 1 more among unvisited
                        if avail < 1:
                            feasible = False
                            break

                if not feasible:
                    continue

                visited_set.add(nb)
                edges_used_local.add(e)
                path.append(nb)

                result = dfs(nb)

                if result is not None:
                    return result

                # Backtrack
                path.pop()
                visited_set.remove(nb)
                edges_used_local.remove(e)

            return None

        return dfs(start)

    best_cycles = []

    # Try to find 3 cycles using multiple strategies
    # Strategy: try different start vertices for the first cycle
    for start_v in range(n):
        if time.time() > deadline - 10:
            break

        used_edges = set()
        cycles = []

        for cyc_idx in range(3):
            if time.time() > deadline - 5:
                break

            # For subsequent cycles, try different starts to improve success rate
            sv = start_v if cyc_idx == 0 else (start_v + cyc_idx * 3) % n
            cycle = find_ham_cycle(used_edges, start=sv)
            if cycle is None:
                break

            ce = get_cycle_edges(cycle)
            used_edges |= ce
            cycles.append(cycle)

        if len(cycles) > len(best_cycles):
            best_cycles = list(cycles)

        if len(best_cycles) >= 3:
            break

    # If we didn't find 3 with the above, try a second round with different ordering
    if len(best_cycles) < 3 and time.time() < deadline - 10:
        for start_v in range(n, 0, -1):  # reverse order
            if time.time() > deadline - 10:
                break

            used_edges = set()
            cycles = []

            for cyc_idx in range(3):
                if time.time() > deadline - 5:
                    break

                sv = start_v if cyc_idx == 0 else (start_v + cyc_idx * 5) % n
                cycle = find_ham_cycle(used_edges, start=sv)
                if cycle is None:
                    break

                ce = get_cycle_edges(cycle)
                used_edges |= ce
                cycles.append(cycle)

            if len(cycles) > len(best_cycles):
                best_cycles = list(cycles)

            if len(best_cycles) >= 3:
                break

    if best_cycles:
        result = []
        for c in best_cycles:
            result.append([int(v) for v in c])
        return result

    # Fallback
    return [[0, 1, 2]]
