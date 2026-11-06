import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def build_adj(edge_set):
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def remove_edge(adj, edge_set, u, v):
        e = tuple(sorted((u, v)))
        edge_set.discard(e)
        adj[u].discard(v)
        adj[v].discard(u)

    def add_edge(adj, edge_set, u, v):
        e = tuple(sorted((u, v)))
        edge_set.add(e)
        adj[u].add(v)
        adj[v].add(u)

    def find_longest_cycle_greedy(adj, edge_set, max_time=5.0):
        """Find a long cycle using greedy DFS with random restarts."""
        if not edge_set:
            return None

        start_time_local = time.time()
        best_cycle = None
        best_len = 0

        vertices = list(adj.keys())
        rng = random.Random(int(time.time() * 1000) % 100000)

        for restart in range(50):
            if time.time() - start_time_local > max_time:
                break
            if not edge_set:
                break

            # Randomly choose starting vertex
            start = rng.choice(vertices)
            if not adj[start]:
                continue

            # Greedy DFS: always go to unvisited neighbor with highest degree
            path = [start]
            visited = {start}
            current = start

            while True:
                # Get unvisited neighbors, sorted by degree descending
                neighbors = [n for n in adj[current] if n not in visited]
                if not neighbors:
                    break

                # Sort by degree descending (greedy)
                neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                next_v = neighbors[0]
                path.append(next_v)
                visited.add(next_v)
                current = next_v

            # Check if path forms a cycle (last vertex connected to first)
            if len(path) >= 3 and adj[path[-1]] and path[0] in adj[path[-1]]:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = list(path)

            # Also try: if path ends at a vertex connected to some earlier vertex
            # (not just the start), we can form a cycle
            if len(path) >= 3:
                for i in range(len(path) - 2, -1, -1):
                    if path[i] in adj[path[-1]]:
                        cycle = path[i:]
                        if len(cycle) >= 3 and len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = list(cycle)
                        break

        return best_cycle

    def find_longest_cycle_bfs(adj, edge_set, max_time=5.0):
        """Find longest cycle by trying BFS from each edge's endpoints."""
        if not edge_set:
            return None

        start_time_local = time.time()
        best_cycle = None
        best_len = 0

        edges_list = list(edge_set)
        rng = random.Random(int(time.time() * 1000) % 100000)
        rng.shuffle(edges_list)

        for u, v in edges_list:
            if time.time() - start_time_local > max_time:
                break
            if (u, v) not in edge_set and (v, u) not in edge_set:
                continue

            # Temporarily remove edge (u,v)
            remove_edge(adj, edge_set, u, v)

            # BFS from u to find longest path to v
            # Actually, let's do a DFS to find a long path
            # Use iterative deepening or just greedy
            path = _find_long_path(adj, u, v, max_depth=20)

            # Restore edge
            add_edge(adj, edge_set, u, v)

            if path and len(path) >= 3:
                if len(path) > best_len:
                    best_len = len(path)
                    best_cycle = list(path)

        return best_cycle

    def _find_long_path(adj, start, end, max_depth=20):
        """Find a long path from start to end using DFS."""
        best_path = None
        best_len = 0

        def dfs(node, path, visited):
            nonlocal best_path, best_len
            if node == end and len(path) >= 3:
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
                return

            if len(path) >= max_depth:
                return

            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        dfs(start, [start], {start})
        return best_path

    def decompose_graph(edges, time_budget):
        """Decompose graph into cycles and edges, minimizing number of pieces."""
        if not edges:
            return []

        best_pieces = None
        best_count = float('inf')

        # Strategy 1: Greedy longest cycle extraction
        pieces1 = _decompose_greedy_longest(edges, time_budget * 0.4)
        if pieces1 is not None:
            best_pieces = pieces1
            best_count = len(pieces1)

        # Strategy 2: Remove low-degree vertices first, then extract cycles
        pieces2 = _decompose_low_degree_first(edges, time_budget * 0.3)
        if pieces2 is not None and len(pieces2) < best_count:
            best_pieces = pieces2
            best_count = len(pieces2)

        # Strategy 3: Randomized greedy with multiple restarts
        pieces3 = _decompose_randomized(edges, time_budget * 0.3)
        if pieces3 is not None and len(pieces3) < best_count:
            best_pieces = pieces3
            best_count = len(pieces3)

        if best_pieces is None:
            return [list(e) for e in edges]

        # Post-processing: try to merge leftover edges into cycles
        best_pieces = _post_process(best_pieces, edges)

        return best_pieces

    def _decompose_greedy_longest(edges, time_budget):
        """Greedy: repeatedly find longest cycle, remove it."""
        start_time_local = time.time()
        remaining = set(edges)
        adj = build_adj(remaining)
        pieces = []

        while remaining:
            if time.time() - start_time_local > time_budget:
                break

            cycle = find_longest_cycle_greedy(adj, remaining, max_time=min(2.0, time_budget - (time.time() - start_time_local)))

            if cycle:
                pieces.append(cycle)
                # Remove edges of cycle
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    remove_edge(adj, remaining, u, v)
                u, v = cycle[-1], cycle[0]
                remove_edge(adj, remaining, u, v)
            else:
                # No more cycles; add remaining edges as single pieces
                for e in list(remaining):
                    pieces.append(list(e))
                    remove_edge(adj, remaining, e[0], e[1])
                break

        return pieces

    def _decompose_low_degree_first(edges, time_budget):
        """Remove low-degree vertices first, then extract cycles from remaining."""
        start_time_local = time.time()
        remaining = set(edges)
        adj = build_adj(remaining)
        pieces = []

        # Phase 1: Remove edges incident to vertices with degree <= 1
        changed = True
        while changed and time.time() - start_time_local < time_budget * 0.3:
            changed = False
            for v in list(adj.keys()):
                if len(adj[v]) <= 1:
                    for u in list(adj[v]):
                        if (u, v) in remaining or (v, u) in remaining:
                            pieces.append([u, v])
                            remove_edge(adj, remaining, u, v)
                            changed = True

        # Phase 2: Extract cycles from remaining
        while remaining and time.time() - start_time_local < time_budget:
            cycle = find_longest_cycle_greedy(adj, remaining, max_time=min(1.0, time_budget - (time.time() - start_time_local)))

            if cycle:
                pieces.append(cycle)
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i + 1]
                    remove_edge(adj, remaining, u, v)
                u, v = cycle[-1], cycle[0]
                remove_edge(adj, remaining, u, v)
            else:
                for e in list(remaining):
                    pieces.append(list(e))
                    remove_edge(adj, remaining, e[0], e[1])
                break

        return pieces

    def _decompose_randomized(edges, time_budget):
        """Randomized greedy with multiple restarts."""
        start_time_local = time.time()
        best_pieces = None
        best_count = float('inf')
        rng = random.Random(42)

        num_restarts = 5
        per_restart_budget = time_budget / num_restarts

        for restart in range(num_restarts):
            if time.time() - start_time_local > time_budget:
                break

            remaining = set(edges)
            adj = build_adj(remaining)
            pieces = []
            local_start = time.time()

            while remaining and time.time() - local_start < per_restart_budget:
                # Randomly pick an edge to start cycle search
                edge_list = list(remaining)
                u, v = rng.choice(edge_list)

                # Try to find a cycle containing this edge
                remove_edge(adj, remaining, u, v)
                path = _find_long_path(adj, u, v, max_depth=15)
                add_edge(adj, remaining, u, v)

                if path and len(path) >= 3:
                    pieces.append(path)
                    for i in range(len(path) - 1):
                        a, b = path[i], path[i + 1]
                        remove_edge(adj, remaining, a, b)
                    a, b = path[-1], path[0]
                    remove_edge(adj, remaining, a, b)
                else:
                    # Can't form cycle through this edge; take it as single edge
                    pieces.append([u, v])
                    remove_edge(adj, remaining, u, v)

            if len(pieces) < best_count:
                best_count = len(pieces)
                best_pieces = pieces

        return best_pieces

    def _post_process(pieces, original_edges):
        """Try to merge leftover single edges into cycles."""
        # Identify single edges
        singles = [p for p in pieces if len(p) == 2]
        cycles = [p for p in pieces if len(p) >= 3]

        if not singles:
            return pieces

        # Build adjacency from all edges
        all_adj = build_adj(set(original_edges))

        # Try to find cycles among single edges
        single_edge_set = {tuple(sorted(p)) for p in singles}
        single_adj = build_adj(single_edge_set)

        improved = False
        new_singles = []
        for e in singles:
            u, v = e
            # Check if there's a path from u to v through other single edges
            # forming a cycle of length >= 3
            # Simple check: is there a neighbor w of u (in single_adj) such that
            # w is also a neighbor of v?
            found_cycle = None
            for w in single_adj.get(u, set()):
                if w == v:
                    continue
                if v in single_adj.get(w, set()):
                    # Cycle: u -> w -> v -> u
                    # But we need to make sure these edges are all in singles
                    if (u, w) in single_edge_set or (w, u) in single_edge_set:
                        if (w, v) in single_edge_set or (v, w) in single_edge_set:
                            found_cycle = [u, w, v]
                            break

            if found_cycle:
                # Remove these three edges from singles
                for i in range(len(found_cycle) - 1):
                    a, b = found_cycle[i], found_cycle[i + 1]
                    single_edge_set.discard(tuple(sorted((a, b))))
                a, b = found_cycle[-1], found_cycle[0]
                single_edge_set.discard(tuple(sorted((a, b))))

                # Update single_adj
                for i in range(len(found_cycle) - 1):
                    a, b = found_cycle[i], found_cycle[i + 1]
                    single_adj[a].discard(b)
                    single_adj[b].discard(a)
                a, b = found_cycle[-1], found_cycle[0]
                single_adj[a].discard(b)
                single_adj[b].discard(a)

                cycles.append(found_cycle)
                improved = True
            else:
                new_singles.append(e)

        if not improved:
            return pieces

        # Rebuild singles list
        result = cycles + new_singles
        return result

    results = []
    for i, edges in enumerate(graphs):
        # Allocate time budget per graph
        elapsed = time.time() - start_time
        remaining_time = deadline - 2.0 - elapsed
        if remaining_time <= 0:
            results.append([list(e) for e in edges])
            continue

        # Distribute time: more time for larger graphs
        if i == 0:
            budget = remaining_time * 0.3
        elif i == 1:
            budget = remaining_time * 0.35
        else:
            budget = remaining_time * 0.35

        pieces = decompose_graph(edges, budget)
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
