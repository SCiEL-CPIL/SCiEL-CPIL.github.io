import time
import random
from collections import defaultdict, deque
from random import Random

_DEADLINE = 140.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def construct():
    def instance_graph(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _validate(paths, edges):
        if not paths:
            return len(edges) == 0
        used = set()
        for p in paths:
            if len(p) < 2:
                return False
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i+1])))
                if e in used:
                    return False
                used.add(e)
        return used == set(edges)

    def _line_graph_matching_decompose(edges, n, max_time=4.0):
        """
        Maximum matching in the line graph of G.
        Each edge of G is a vertex in L(G). Two are adjacent if they share a vertex.
        A matching pairs consecutive edges in paths. Unmatched edges are path endpoints.
        Number of paths = |E| - |matching|.
        """
        if not edges:
            return []
        if _remaining() < 1.0:
            return [[u, v] for u, v in edges]

        edge_list = sorted(set(tuple(sorted(e)) for e in edges))
        m = len(edge_list)
        if m == 0:
            return []

        # Build line graph adjacency: edge i adjacent to edge j if they share a vertex
        # For each vertex, collect incident edge indices
        vertex_edges = defaultdict(list)
        for i, (u, v) in enumerate(edge_list):
            vertex_edges[u].append(i)
            vertex_edges[v].append(i)

        adj_line = [set() for _ in range(m)]
        for v, idxs in vertex_edges.items():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    adj_line[i].add(j)
                    adj_line[j].add(i)

        # Greedy matching with local improvement (2-opt / swap / augmenting)
        best_match = [-1] * m
        best_size = 0
        start_time = time.time()

        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() - start_time > max_time or _remaining() < 3.0:
                break

            match = [-1] * m
            matched = [False] * m

            # Greedy: process edges in random order, pick most constrained neighbor
            order = list(range(m))
            random.shuffle(order)
            for i in order:
                if matched[i]:
                    continue
                candidates = [j for j in adj_line[i] if not matched[j]]
                if candidates:
                    best_j = min(candidates, key=lambda j: sum(1 for k in adj_line[j] if not matched[k]))
                    match[i] = best_j
                    match[best_j] = i
                    matched[i] = True
                    matched[j if False else best_j] = True

            size = sum(1 for i in range(m) if match[i] != -1) // 2

            # Local improvement: try augmenting paths and swaps
            improved = True
            while improved and time.time() - start_time < max_time and _remaining() > 3.0:
                improved = False
                # Try 1-flip: unmatched edge i, find j unmatched neighbor -> add
                for i in range(m):
                    if match[i] != -1:
                        continue
                    for j in adj_line[i]:
                        if match[j] == -1:
                            match[i] = j
                            match[j] = i
                            size += 1
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue

                # Try 2-opt swap: matched i-j, find k unmatched neighbor of i, l unmatched neighbor of j
                # Break i-j, make i-k and j-l (if k and l exist and are distinct)
                for i in range(m):
                    if match[i] == -1:
                        continue
                    j = match[i]
                    for k in adj_line[i]:
                        if k == j or match[k] != -1:
                            continue
                        for l in adj_line[j]:
                            if l == i or l == k or match[l] != -1:
                                continue
                            match[i] = k
                            match[k] = i
                            match[j] = l
                            match[l] = j
                            size += 1
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    continue

                # Try augmenting path of length 3: unmatched a - matched(b,c) - unmatched d
                # where a adj b, d adj c, and a!=d. Replace b-c with a-b and c-d.
                for a in range(m):
                    if match[a] != -1:
                        continue
                    for b in adj_line[a]:
                        if match[b] == -1:
                            continue
                        c = match[b]
                        for d in adj_line[c]:
                            if d == a or match[d] != -1:
                                continue
                            match[a] = b
                            match[b] = a
                            match[c] = d
                            match[d] = c
                            size += 1
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break

            if size > best_size:
                best_size = size
                best_match = match[:]

        # Reconstruct paths from matching
        if best_size == 0:
            return [[u, v] for u, v in edge_list]

        # For each edge, determine which vertex connects it to its matched neighbor
        # Build a directed structure: for matched pair (i, j), they share a vertex v.
        # Edge i = (u_i, v_i), edge j = (u_j, v_j). Shared vertex is the connection point.
        # We build adjacency: edge i -> (shared_vertex, edge j)
        edge_to_shared = {}
        for i in range(m):
            j = best_match[i]
            if j == -1 or j <= i:
                continue
            u_i, v_i = edge_list[i]
            u_j, v_j = edge_list[j]
            # Find shared vertex
            shared = None
            if u_i == u_j:
                shared = u_i
            elif u_i == v_j:
                shared = u_i
            elif v_i == u_j:
                shared = v_i
            elif v_i == v_j:
                shared = v_i
            if shared is None:
                continue
            edge_to_shared[(i, j)] = shared
            edge_to_shared[(j, i)] = shared

        # Now trace paths. Each edge has at most 2 matched neighbors (one at each endpoint).
        # Actually in a matching, each edge is matched to at most 1 other edge.
        # So the structure is a collection of paths where consecutive edges share a vertex.
        # But wait: a matching means each edge is in at most one pair. So the "paths" are
        # just pairs of edges plus singletons? That gives |E| - |matching| paths of length 2 or 1.
        #
        # That's not right for minimizing paths. The issue: a matching in the line graph
        # only pairs edges, it doesn't chain them. To get long paths, we need a path cover
        # where each edge is adjacent to up to 2 others (one on each side).
        #
        # A proper path cover in the line graph is a set of vertex-disjoint paths covering
        # all vertices of L(G). This is equivalent to a 2-matching (each vertex degree <= 2,
        # no cycles unless it's the whole graph) in L(G). The number of paths = |V(L)| - |E(2-matching)|.
        #
        # So we need a maximum 2-matching (also called a cycle cover with paths), not a 1-matching.
        # Let me redo: find a maximum set of edges in L(G) such that each vertex has degree <= 2
        # and there are no cycles (or cycles are allowed if we can break them).
        #
        # Actually, Gallai's theorem: min paths = max(1, |odd|/2). The line graph approach
        # with 2-matching: a 2-matching in L(G) with no cycles gives a path decomposition.
        # Maximize the number of edges in the 2-matching to minimize paths.

        # Let me implement a 2-matching (degree-constrained subgraph) instead.
        return None  # Signal to use the 2-matching approach below

    def _two_matching_decompose(edges, n, max_time=5.0):
        """
        Find a maximum 2-matching in the line graph (each vertex degree <= 2, no cycles).
        This gives a path decomposition. Number of paths = |E| - |2-matching edges|.
        """
        if not edges:
            return []
        if _remaining() < 1.0:
            return [[u, v] for u, v in edges]

        edge_list = sorted(set(tuple(sorted(e)) for e in edges))
        m = len(edge_list)
        if m == 0:
            return []

        # Build line graph adjacency
        vertex_edges = defaultdict(list)
        for i, (u, v) in enumerate(edge_list):
            vertex_edges[u].append(i)
            vertex_edges[v].append(i)

        adj_line = [set() for _ in range(m)]
        for v, idxs in vertex_edges.items():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    adj_line[i].add(j)
                    adj_line[j].add(i)

        # 2-matching: select edges from L(G) such that each vertex has degree <= 2, no cycles.
        # Greedy: build a set of selected adjacencies, each edge i can be connected to up to 2 others.
        # We want to maximize total selected adjacencies.

        best_selected = None
        best_count = 0
        start_time = time.time()

        max_restarts = 300
        for restart in range(max_restarts):
            if time.time() - start_time > max_time or _remaining() < 3.0:
                break

            # selected[i] = set of neighbors of i in the 2-matching (size <= 2)
            selected = [set() for _ in range(m)]
            count = 0

            # Greedy: process edges in random order, connect to up to 2 available neighbors
            order = list(range(m))
            random.shuffle(order)
            for i in order:
                if len(selected[i]) >= 2:
                    continue
                # Find available neighbors (degree < 2 in selected)
                candidates = [j for j in adj_line[i] if len(selected[j]) < 2]
                if not candidates:
                    continue
                # Pick the most constrained (fewest remaining options)
                # But we can add up to (2 - len(selected[i])) edges
                slots = 2 - len(selected[i])
                candidates.sort(key=lambda j: sum(1 for k in adj_line[j] if len(selected[k]) < 2))
                for j in candidates[:slots]:
                    if len(selected[i]) >= 2:
                        break
                    if len(selected[j]) >= 2:
                        continue
                    selected[i].add(j)
                    selected[j].add(i)
                    count += 1

            # Check for cycles and break them
            # A cycle in the 2-matching means all vertices in the cycle have degree 2.
            # We can break a cycle by removing one edge, reducing count by 1 but making it a path.
            # Actually, a cycle in L(G) corresponds to a cycle in the path decomposition,
            # which we can break at any point to form a path. So cycles are OK - we just
            # need to convert them to paths. The count of "paths" is the number of connected
            # components in the 2-matching (each component is a path or cycle).
            # For a cycle, it's 1 component = 1 path (after breaking).
            # So number of paths = number of connected components in the 2-matching.

            # Count connected components
            visited = [False] * m
            components = 0
            for i in range(m):
                if not visited[i] and selected[i]:
                    components += 1
                    stack = [i]
                    visited[i] = True
                    while stack:
                        u = stack.pop()
                        for v in selected[u]:
                            if not visited[v]:
                                visited[v] = True
                                stack.append(v)
            # Edges with no selected neighbors are isolated components (single-edge paths)
            for i in range(m):
                if not visited[i]:
                    components += 1
                    visited[i] = True

            # We want to minimize components = maximize (m - components) = maximize count
            # But components is what matters. Let's track components directly.
            # Actually, for a 2-matching with no isolated vertices, components = m - count.
            # With isolated vertices, components = (isolated) + (m - isolated - count).
            # Let's just compute components.

            if count > best_count:
                best_count = count
                best_selected = [set(s) for s in selected]

            # Local improvement: try to add edges
            improved = True
            while improved and time.time() - start_time < max_time and _remaining() > 3.0:
                improved = False
                for i in range(m):
                    if len(selected[i]) >= 2:
                        continue
                    candidates = [j for j in adj_line[i] if len(selected[j]) < 2 and j not in selected[i]]
                    if candidates:
                        j = min(candidates, key=lambda j: len(selected[j]))
                        selected[i].add(j)
                        selected[j].add(i)
                        count += 1
                        improved = True
                        break

            if count > best_count:
                best_count = count
                best_selected = [set(s) for s in selected]

        if best_selected is None or best_count == 0:
            return [[u, v] for u, v in edge_list]

        # Reconstruct paths from the 2-matching
        # Each edge i has up to 2 neighbors in the 2-matching.
        # We need to determine the direction: for edge i = (u, v) and neighbor j,
        # they share a vertex. The path goes through that shared vertex.

        # Build the path structure: for each edge, determine its two endpoints in the path.
        # An edge i = (u, v) connects to neighbor j1 at vertex u and neighbor j2 at vertex v (or vice versa).

        # For each edge, determine which vertex connects to which neighbor
        edge_connections = {}  # edge i -> (vertex_to_j1, j1, vertex_to_j2, j2) or similar
        for i in range(m):
            u_i, v_i = edge_list[i]
            nbrs = list(selected[i])
            if len(nbrs) == 0:
                edge_connections[i] = (None, None, None, None)
            elif len(nbrs) == 1:
                j = nbrs[0]
                u_j, v_j = edge_list[j]
                if u_i == u_j or u_i == v_j:
                    edge_connections[i] = (u_i, j, None, None)
                else:
                    edge_connections[i] = (v_i, j, None, None)
            else:
                j1, j2 = nbrs
                u_j1, v_j1 = edge_list[j1]
                u_j2, v_j2 = edge_list[j2]
                # Determine which vertex of i connects to j1
                if u_i == u_j1 or u_i == v_j1:
                    v1 = u_i
                else:
                    v1 = v_i
                if u_i == u_j2 or u_i == v_j2:
                    v2 = u_i
                else:
                    v2 = v_i
                edge_connections[i] = (v1, j1, v2, j2)

        # Now trace paths. Find endpoints: edges with only 0 or 1 neighbor.
        # Actually, edges with 1 neighbor are endpoints of a path (degree 1 in 2-matching).
        # Edges with 0 neighbors are isolated (single-edge paths).
        # Edges with 2 neighbors are internal.

        # Build a directed graph: from edge i, go to neighbor j through the shared vertex.
        # We need to orient: edge i = (a, b), if it connects to j at vertex a, then the path
        # enters i at a and leaves at b (or vice versa depending on the other connection).

        # Let's build adjacency: for each edge i, its two "ports" (vertices) and which edge connects to each.
        # Then trace from endpoints.

        # For each edge, determine the two vertices and which neighbor is at each vertex.
        edge_ports = {}  # edge i -> {vertex1: neighbor1, vertex2: neighbor2}
        for i in range(m):
            u_i, v_i = edge_list[i]
            nbrs = list(selected[i])
            ports = {}
            for j in nbrs:
                u_j, v_j = edge_list[j]
                if u_i == u_j or u_i == v_j:
                    ports[u_i] = j
                else:
                    ports[v_i] = j
            edge_ports[i] = ports

        # Find path endpoints: edges with degree 1 in the 2-matching
        # Or isolated edges (degree 0)
        visited_edges = set()
        paths = []

        # Start from edges with degree 1 (endpoints of paths)
        for i in range(m):
            if i in visited_edges:
                continue
            if len(selected[i]) == 0:
                # Isolated edge
                paths.append([edge_list[i][0], edge_list[i][1]])
                visited_edges.add(i)
            elif len(selected[i]) == 1:
                # Start of a path
                path = []
                current = i
                prev = None
                while current is not None and current not in visited_edges:
                    visited_edges.add(current)
                    u, v = edge_list[current]
                    # Determine which vertex we entered from
                    if prev is None:
                        # We're at an endpoint. The path starts at the vertex not connected to a neighbor.
                        # The connected vertex is the one in edge_ports
                        conn_vertex = list(edge_ports[current].keys())[0]
                        if conn_vertex == u:
                            path.append(v)  # Start at the free end
                        else:
                            path.append(u)
                    else:
                        # We came from prev through a shared vertex
                        # The shared vertex is the one that connects current to prev
                        if prev in edge_ports[current].values():
                            shared = [k for k, val in edge_ports[current].items() if val == prev][0]
                            # We entered current at 'shared', so we leave at the other vertex
                            other = v if shared == u else u
                            if not path:
                                path.append(shared)
                            path.append(other)
                        else:
                            # Should not happen
                            break

                    # Move to the next edge
                    if len(selected[current]) >= 2:
                        # Go to the other neighbor
                        nbrs = list(selected[current])
                        next_edge = None
                        for j in nbrs:
                            if j != prev:
                                next_edge = j
                                break
                        prev = current
                        current = next_edge
                    else:
                        # Degree 1, this is the end
                        current = None

                if len(path) >= 2:
                    paths.append(path)

        # Check for any unvisited edges (cycles)
        for i in range(m):
            if i not in visited_edges:
                # Cycle
                path = []
                current = i
                prev = None
                start = i
                while current is not None and current != start:
                    visited_edges.add(current)
                    u, v = edge_list[current]
                    if prev is None:
                        conn_vertex = list(edge_ports[current].keys())[0]
                        if conn_vertex == u:
                            path.append(v)
                        else:
                            path.append(u)
                    else:
                        if prev in edge_ports[current].values():
                            shared = [k for k, val in edge_ports[current].items() if val == prev][0]
                            other = v if shared == u else u
                            if not path:
                                path.append(shared)
                            path.append(other)
                        else:
                            break

                    nbrs = list(selected[current])
                    next_edge = None
                    for j in nbrs:
                        if j != prev and j != start:
                            next_edge = j
                            break
                    if next_edge is None:
                        # Check if we can go to start
                        if start in nbrs and start != prev:
                            next_edge = start
                        else:
                            next_edge = None

                    prev = current
                    current = next_edge

                if len(path) >= 2:
                    paths.append(path)

        # Validate
        if _validate(paths, edge_list):
            return paths

        return None

    def _greedy_decompose(edges, n, max_time=3.0):
        """Greedy path decomposition starting from odd vertices."""
        if not edges:
            return []
        if _remaining() < 1.0:
            return [[u, v] for u, v in edges]

        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        best_paths = None
        best_count = float('inf')
        start_time = time.time()

        while time.time() - start_time < max_time and _remaining() > 2.0:
            remaining = set(edge_set)
            paths = []

            deg = {i: 0 for i in range(n)}
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]

            while remaining:
                rem_deg = {i: 0 for i in range(n)}
                for u, v in remaining:
                    rem_deg[u] += 1
                    rem_deg[v] += 1
                rem_odd = [v for v in rem_deg if rem_deg[v] > 0 and rem_deg[v] % 2 == 1]

                if rem_odd:
                    start_v = random.choice(rem_odd)
                else:
                    candidates = [v for v in rem_deg if rem_deg[v] > 0]
                    if not candidates:
                        break
                    start_v = random.choice(candidates)

                path = _extend_path(start_v, remaining, adj)

                if not path or len(path) < 2:
                    for u, v in remaining:
                        if u == start_v:
                            path = [start_v, v]
                            remaining.discard((u, v))
                            break
                        elif v == start_v:
                            path = [start_v, u]
                            remaining.discard((u, v))
                            break
                    if not path:
                        break
                else:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        e = tuple(sorted((u, v)))
                        remaining.discard(e)

                paths.append(path)

            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]

        if best_paths is None:
            return [[u, v] for u, v in sorted(edge_set)]
        return best_paths

    def _extend_path(start, remaining, adj):
        path = [start]
        visited = {start}
        current = start

        while True:
            available = []
            for nb in adj[current]:
                e = tuple(sorted((current, nb)))
                if e in remaining and nb not in visited:
                    deg = 0
                    for x in adj[nb]:
                        e2 = tuple(sorted((nb, x)))
                        if e2 in remaining and x not in visited:
                            deg += 1
                    available.append((deg, nb))

            if not available:
                break

            available.sort()
            min_deg = available[0][0]
            candidates = [nb for deg, nb in available if deg == min_deg]
            best_nb = random.choice(candidates)

            e = tuple(sorted((current, best_nb)))
            remaining.discard(e)
            path.append(best_nb)
            visited.add(best_nb)
            current = best_nb

        return path

    def _solve_one(edges, n, max_time=6.0):
        """Try 2-matching first, fall back to greedy."""
        if not edges:
            return []

        # Try 2-matching approach
        paths = _two_matching_decompose(edges, n, max_time=max_time * 0.6)
        if paths is not None and _validate(paths, edges):
            return paths

        # Fall back to greedy
        paths = _greedy_decompose(edges, n, max_time=max_time * 0.4)
        if _validate(paths, edges):
            return paths

        # Last resort: individual edges
        return [[u, v] for u, v in edges]

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []

    for i, (n, p, seed) in enumerate(specs):
        edges = instance_graph(n, p, seed)
        max_t = 8.0
        if _remaining() < max_t:
            max_t = max(1.0, _remaining() - 2.0)

        paths = _solve_one(edges, n, max_time=max_t)
        results.append(paths)

    return results

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
