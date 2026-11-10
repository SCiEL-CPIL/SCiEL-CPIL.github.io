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

    def min_cost_flow(s, t, graph, max_flow):
        """
        Min-cost max-flow using successive shortest paths with potentials.
        graph: dict of dict: graph[u][v] = [capacity, cost]
        Returns (flow, cost)
        """
        import heapq

        n = len(graph)
        # Add reverse edges if not present
        for u in graph:
            for v in graph[u]:
                if v not in graph:
                    graph[v] = {}
                if u not in graph[v]:
                    graph[v][u] = [0, 0]

        potential = [0] * n
        flow = 0
        cost = 0

        while flow < max_flow:
            # Dijkstra with potentials
            dist = [float('inf')] * n
            prev = [-1] * n
            dist[s] = 0
            pq = [(0, s)]

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, (cap, c) in graph[u].items():
                    if cap > 0:
                        nd = d + c + potential[u] - potential[v]
                        if nd < dist[v]:
                            dist[v] = nd
                            prev[v] = u
                            heapq.heappush(pq, (nd, v))

            if dist[t] == float('inf'):
                break

            # Update potentials
            for i in range(n):
                if dist[i] < float('inf'):
                    potential[i] += dist[i]

            # Find augmenting path
            v = t
            path_flow = max_flow - flow
            while v != s:
                u = prev[v]
                path_flow = min(path_flow, graph[u][v][0])
                v = u

            # Update capacities and cost
            v = t
            while v != s:
                u = prev[v]
                graph[u][v][0] -= path_flow
                graph[v][u][0] += path_flow
                cost += path_flow * graph[u][v][1]
                v = u

            flow += path_flow

        return flow, cost

    def solve_with_flow(edges, n):
        """
        Solve path cover using min-cost flow.
        Each undirected edge is split into two directed arcs.
        We want to find a set of vertex-disjoint paths that cover all edges.
        Actually, we want edge-disjoint paths that cover all edges.
        
        Formulation:
        - Create a directed graph where each undirected edge (u,v) becomes two arcs (u->v) and (v->u) with capacity 1 and cost 0.
        - We want to decompose the graph into paths.
        - A path decomposition corresponds to a set of directed paths that cover all arcs exactly once.
        - This is equivalent to finding a minimum number of paths.
        
        Alternative formulation using flow:
        - For each vertex v, we have in-degree and out-degree in the path decomposition.
        - A vertex is an endpoint of a path if in-degree != out-degree.
        - Specifically, if out-degree > in-degree, it's a start; if in-degree > out-degree, it's an end.
        - For each vertex, |out_deg - in_deg| is either 0 (internal or isolated) or 1 (endpoint).
        - The number of paths = sum of (out_deg - in_deg) over all vertices = (number of endpoints) / 2.
        
        We can formulate this as:
        - Choose a direction for each edge.
        - Minimize the number of paths.
        
        This is a complex combinatorial problem. Let's try a different approach.
        
        Actually, let's use a simpler heuristic first, and then try to improve.
        """
        # For now, use the greedy approach but with better heuristics
        pass

    def greedy_path_cover_improved(edges, n, rng):
        """
        Improved greedy path cover with better heuristics.
        """
        if not edges:
            return []

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []

        # Build a list of vertices sorted by degree
        vertices = list(adj.keys())
        
        while remaining_edges:
            # Find a start vertex: prefer degree 1, then degree 2, then higher
            degree1 = [u for u in adj if len(adj[u]) == 1]
            degree2 = [u for u in adj if len(adj[u]) == 2]
            higher = [u for u in adj if len(adj[u]) > 2]

            if degree1:
                start = rng.choice(degree1)
            elif degree2:
                start = rng.choice(degree2)
            elif higher:
                start = rng.choice(higher)
            else:
                break

            path = [start]
            visited_vertices = {start}
            current = start

            while True:
                # Get valid neighbors
                valid_neighbors = []
                for neighbor in adj[current]:
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in remaining_edges and neighbor not in visited_vertices:
                        valid_neighbors.append(neighbor)

                if not valid_neighbors:
                    break

                # Choose next vertex with lookahead
                if len(valid_neighbors) == 1:
                    next_v = valid_neighbors[0]
                else:
                    # Score each neighbor: prefer those with fewer options (to avoid dead ends later)
                    # Actually, prefer those with more options to extend the path
                    def score(neighbor):
                        unvisited_count = 0
                        for nb in adj[neighbor]:
                            if nb not in visited_vertices and nb != current:
                                edge = (min(neighbor, nb), max(neighbor, nb))
                                if edge in remaining_edges:
                                    unvisited_count += 1
                        return unvisited_count

                    # Prefer higher score (more options ahead)
                    best_score = -1
                    best_candidates = []
                    for v in valid_neighbors:
                        s = score(v)
                        if s > best_score:
                            best_score = s
                            best_candidates = [v]
                        elif s == best_score:
                            best_candidates.append(v)
                    next_v = rng.choice(best_candidates)

                path.append(next_v)
                visited_vertices.add(next_v)
                current = next_v

            if len(path) > 1:
                paths.append(path)

            # Remove edges used in this path
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    adj[u].remove(v)
                    adj[v].remove(u)

            if time.time() > deadline:
                break

        return paths

    def merge_paths(paths, edges, n):
        """
        Try to merge paths by finding connecting edges.
        """
        if len(paths) < 2:
            return paths

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Build a set of used edges
        used_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = (min(u, v), max(u, v))
                used_edges.add(edge)

        all_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        remaining_edges = all_edges - used_edges

        # Try to merge paths
        improved = True
        while improved:
            improved = False
            for i in range(len(paths)):
                for j in range(len(paths)):
                    if i == j:
                        continue

                    # Try connecting end of path i to start of path j
                    end_i = paths[i][-1]
                    start_j = paths[j][0]
                    edge = (min(end_i, start_j), max(end_i, start_j))
                    if edge in remaining_edges:
                        if end_i not in set(paths[j]):
                            paths[i] = paths[i] + paths[j]
                            remaining_edges.remove(edge)
                            paths.pop(j)
                            improved = True
                            break

                    # Try connecting start of path i to end of path j
                    start_i = paths[i][0]
                    end_j = paths[j][-1]
                    edge = (min(start_i, end_j), max(start_i, end_j))
                    if edge in remaining_edges:
                        if end_j not in set(paths[i]):
                            paths[i] = list(reversed(paths[j])) + paths[i]
                            remaining_edges.remove(edge)
                            paths.pop(j)
                            improved = True
                            break

                if improved:
                    break

        return paths

    def solve_graph(edges, n):
        best_cover = None
        best_count = float('inf')

        # Run greedy with different seeds
        for seed in range(30):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            cover = greedy_path_cover_improved(edges, n, rng)
            cover = merge_paths(cover, edges, n)
            count = len(cover)
            if count < best_count:
                best_count = count
                best_cover = cover

        if best_cover is None:
            best_cover = [[list(e)] for e in edges]

        return best_cover

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        edges = instance(*spec)
        n = spec[0]
        cover = solve_graph(edges, n)
        results.append(cover)

    return results
