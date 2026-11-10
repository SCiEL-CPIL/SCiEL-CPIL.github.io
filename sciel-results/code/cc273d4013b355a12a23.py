"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from random import Random

def construct():
    _start = time.time()
    _DEADLINE = 140.0  # 10s safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def greedy_path_cover(edges):
        """
        Greedy edge-disjoint path cover:
        1. Build adjacency with edge IDs.
        2. Repeatedly find a longest simple path in the remaining edge graph
           using a heuristic (start from a vertex with highest remaining degree,
           then do a greedy walk preferring neighbors with higher remaining degree).
        3. Remove used edges, repeat until all edges are covered.
        """
        if not edges:
            return []

        # Map each undirected edge to an index
        edge_set = set()
        for e in edges:
            u, v = e
            edge_set.add((u, v)) if u <= v else edge_set.add((v, u))

        adj = defaultdict(list)  # adj[u] = list of (v, edge_key)
        for (u, v) in edge_set:
            adj[u].append((v, (u, v)))
            adj[v].append((u, (u, v)))

        used_edges = set()
        paths = []

        while edge_set - used_edges:
            remaining = edge_set - used_edges
            if not remaining:
                break

            # Build remaining adjacency
            rem_adj = defaultdict(list)
            for (u, v) in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)

            # Choose start vertex: highest degree in remaining graph
            start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x])) if rem_adj else None
            if start is None:
                break

            # Greedy longest path: walk preferring unvisited neighbors with high remaining degree
            path = [start]
            visited_in_path = {start}
            current = start
            while True:
                neighbors = [n for n in rem_adj[current] if n not in visited_in_path]
                if not neighbors:
                    break
                # Prefer neighbor with highest remaining degree (excluding used edges)
                # Count how many remaining edges each neighbor has to unvisited vertices
                best_neighbor = max(neighbors, key=lambda nb: len([x for x in rem_adj[nb] if x not in visited_in_path]))
                path.append(best_neighbor)
                visited_in_path.add(best_neighbor)
                current = best_neighbor

            # Convert path to edge list and mark as used
            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v) if u <= v else (v, u)
                path_edges.add(key)

            # Verify all edges in path are actually in remaining
            valid = all(e in remaining for e in path_edges)
            if valid and path_edges:
                used_edges |= path_edges
                paths.append(path)
            else:
                # Fallback: just take one edge
                e = next(iter(remaining))
                used_edges.add(e)
                paths.append([e[0], e[1]])

        return paths

    def improved_path_cover(edges):
        """
        Better approach: Use the fact that a graph can be decomposed into
        at most (number of odd-degree vertices)/2 + 1 paths if we allow
        adding edges to make it Eulerian, then decompose into trails.
        
        Strategy:
        1. Find all odd-degree vertices.
        2. Pair them up and find shortest paths between pairs.
        3. Add those paths as "virtual edges" to make all degrees even (Eulerian).
        4. Find Eulerian circuits/trails in the augmented multigraph.
        5. Break the Eulerian circuits at the virtual edges to get the original paths.
        
        This gives at most |odd|/2 + 1 paths per connected component.
        """
        if not edges:
            return []

        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Find odd-degree vertices
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]

        if not odd_vertices:
            # Already Eulerian - one circuit
            # Find Eulerian circuit
            circuit = find_eulerian_circuit(edges)
            if circuit:
                return [circuit]
            return greedy_path_cover(edges)

        # Pair odd vertices and find shortest paths
        # Use BFS for shortest paths
        def bfs_shortest_path(src, dst, adj_map):
            if src == dst:
                return []
            visited = {src}
            parent = {src: None}
            q = deque([src])
            while q:
                cur = q.popleft()
                for nb in adj_map[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = cur
                        if nb == dst:
                            # Reconstruct path
                            path = []
                            node = dst
                            while node is not None:
                                path.append(node)
                                node = parent[node]
                            path.reverse()
                            return path
                        q.append(nb)
            return None  # No path found

        # Pair up odd vertices greedily (pair i with i+1)
        # For better results, we could try to minimize total path length
        pairs = []
        for i in range(0, len(odd_vertices) - 1, 2):
            pairs.append((odd_vertices[i], odd_vertices[i + 1]))

        # Find shortest paths for each pair
        virtual_edges = []
        for (a, b) in pairs:
            sp = bfs_shortest_path(a, b, adj)
            if sp and len(sp) >= 2:
                # Add edges of shortest path as virtual
                for i in range(len(sp) - 1):
                    u, v = sp[i], sp[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    virtual_edges.append(key)

        # Now the augmented graph has all even degrees
        # Build multigraph: original edges + virtual edges (with multiplicity)
        # We need to find Eulerian circuit in the augmented multigraph
        # Then extract the path by cutting at virtual edges

        # Build augmented adjacency with edge tracking
        # Each edge in the augmented graph: (u, v, is_virtual)
        aug_adj = defaultdict(list)
        for (u, v) in edges:
            aug_adj[u].append((v, False))
            aug_adj[v].append((u, False))
        for (u, v) in virtual_edges:
            aug_adj[u].append((v, True))
            aug_adj[v].append((u, True))

        # Find Eulerian circuit using Hierholzer's algorithm
        # We track edges by (u, v, is_virtual, instance) to handle multi-edges
        # Simpler: use a dict of remaining edge counts
        edge_count = defaultdict(int)
        for (u, v) in edges:
            key = (u, v) if u <= v else (v, u)
            edge_count[key] += 1
        for (u, v) in virtual_edges:
            key = (u, v) if u <= v else (v, u)
            edge_count[key] += 1

        # Hierholzer's
        # adjacency with edge keys
        hier_adj = defaultdict(list)
        for (u, v) in edges:
            key = (u, v) if u <= v else (v, u)
            hier_adj[u].append((v, key))
            hier_adj[v].append((u, key))
        for (u, v) in virtual_edges:
            key = (u, v) if u <= v else (v, u)
            hier_adj[u].append((v, key))
            hier_adj[v].append((u, key))

        # Track which edges are used (by key + count)
        used_count = defaultdict(int)

        def eulerian_circuit(start):
            stack = [start]
            circuit = []
            while stack:
                v = stack[-1]
                # Find an unused edge from v
                found = False
                for i, (nb, key) in enumerate(hier_adj[v]):
                    if used_count[key] < edge_count[key]:
                        used_count[key] += 1
                        stack.append(nb)
                        found = True
                        break
                if not found:
                    circuit.append(stack.pop())
            circuit.reverse()
            return circuit

        circuit = eulerian_circuit(odd_vertices[0] if odd_vertices else list(hier_adj.keys())[0])

        # Now break the circuit at virtual edges
        # The circuit is a sequence of vertices
        # Virtual edges are those in virtual_edges set
        virtual_set = set()
        for (u, v) in virtual_edges:
            virtual_set.add((u, v) if u <= v else (v, u))

        # Walk through circuit, cut when we traverse a virtual edge
        paths = []
        current_path = [circuit[0]]
        for i in range(1, len(circuit)):
            u, v = circuit[i - 1], circuit[i]
            key = (u, v) if u <= v else (v, u)
            current_path.append(v)
            if key in virtual_set:
                # This edge is virtual - cut here
                if len(current_path) >= 2:
                    paths.append(current_path)
                current_path = [v]

        if len(current_path) >= 2:
            paths.append(current_path)

        # Filter out trivial paths (single vertex)
        paths = [p for p in paths if len(p) >= 2]

        # Verify: check that all original edges are covered
        covered = set()
        for p in paths:
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key in edge_set:  # will check below
                    covered.add(key)

        # Actually need to verify properly
        orig_edge_set = set()
        for (u, v) in edges:
            orig_edge_set.add((u, v) if u <= v else (v, u))

        covered = set()
        for p in paths:
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key in orig_edge_set:
                    covered.add(key)

        if covered == orig_edge_set:
            # Also verify paths are simple and use only graph edges
            valid = True
            for p in paths:
                if len(set(p)) != len(p):
                    valid = False
                    break
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key not in orig_edge_set:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return paths

        # Fallback to greedy
        return greedy_path_cover(edges)

    def find_eulerian_circuit(edges):
        """Find an Eulerian circuit in a graph where all vertices have even degree."""
        adj = defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)

        if not adj:
            return []

        start = next(iter(adj))
        edge_count = defaultdict(int)
        for (u, v) in edges:
            key = (u, v) if u <= v else (v, u)
            edge_count[key] += 1

        hier_adj = defaultdict(list)
        for (u, v) in edges:
            key = (u, v) if u <= v else (v, u)
            hier_adj[u].append((v, key))
            hier_adj[v].append((u, key))

        used_count = defaultdict(int)
        stack = [start]
        circuit = []
        while stack:
            v = stack[-1]
            found = False
            for (nb, key) in hier_adj[v]:
                if used_count[key] < edge_count[key]:
                    used_count[key] += 1
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        circuit.reverse()
        return circuit

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 5:
            # Not enough time, use greedy for remaining
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))
            continue

        edges = instance(*spec)
        try:
            paths = improved_path_cover(edges)
        except Exception:
            paths = greedy_path_cover(edges)

        # Validate
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))

        covered = set()
        valid = True
        for p in paths:
            if len(p) < 2:
                valid = False
                break
            if len(set(p)) != len(p):
                valid = False
                break
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    valid = False
                    break
                if key in covered:
                    valid = False
                    break
                covered.add(key)
            if not valid:
                break

        if not valid or covered != edge_set:
            paths = greedy_path_cover(edges)
            # Re-validate greedy
            covered2 = set()
            valid2 = True
            for p in paths:
                if len(p) < 2:
                    valid2 = False
                    break
                if len(set(p)) != len(p):
                    valid2 = False
                    break
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key not in edge_set:
                        valid2 = False
                        break
                    if key in covered2:
                        valid2 = False
                        break
                    covered2.add(key)
                if not valid2:
                    break
            if not valid2 or covered2 != edge_set:
                # Last resort: each edge as its own path
                paths = [[list(e)] for e in edges]

        results.append(paths)

    return results
# EVOLVE-BLOCK-END
