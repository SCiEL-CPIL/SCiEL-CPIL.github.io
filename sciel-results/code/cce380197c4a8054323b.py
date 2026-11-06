import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def extract_2factor(adj, n, remaining_edges):
        """
        Extract a 2-factor (collection of vertex-disjoint cycles) from the graph
        using max-flow. Returns list of cycles (each as list of vertices).
        """
        if not remaining_edges:
            return []

        # Build flow network
        # Source -> vertex_in, vertex_out -> sink
        # vertex_in -> vertex_out (capacity 1)
        # For each edge (u,v), add edges u_out -> v_in and v_out -> u_in with capacity 1

        source = n * 2
        sink = n * 2 + 1
        num_nodes = n * 2 + 2

        # Adjacency list for flow network
        # Each edge: [to, cap, rev_index]
        flow_adj = [[] for _ in range(num_nodes)]

        def add_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])

        # Source to vertex_in
        for i in range(n):
            add_edge(source, i, 1)  # source -> i (in)

        # vertex_out to sink
        for i in range(n):
            add_edge(i + n, sink, 1)  # i+n (out) -> sink

        # vertex_in -> vertex_out (capacity 1, ensures each vertex used at most once)
        for i in range(n):
            add_edge(i, i + n, 1)

        # For each remaining edge (u,v), add directed edges both ways
        for (u, v) in remaining_edges:
            # u_out -> v_in and v_out -> u_in
            add_edge(u + n, v, 1)
            add_edge(v + n, u, 1)

        # Max flow using Edmonds-Karp (BFS)
        def max_flow():
            total_flow = 0
            while True:
                # BFS to find augmenting path
                parent = [-1] * num_nodes
                parent_edge = [-1] * num_nodes  # index of edge used
                queue = deque([source])
                parent[source] = source
                found = False

                while queue and not found:
                    u = queue.popleft()
                    for idx, (v, cap, _) in enumerate(flow_adj[u]):
                        if cap > 0 and parent[v] == -1:
                            parent[v] = u
                            parent_edge[v] = idx
                            if v == sink:
                                found = True
                                break
                            queue.append(v)

                if not found:
                    break

                # Find bottleneck
                bottleneck = float('inf')
                v = sink
                while v != source:
                    u = parent[v]
                    idx = parent_edge[v]
                    bottleneck = min(bottleneck, flow_adj[u][idx][1])
                    v = u

                # Update residual capacities
                v = sink
                while v != source:
                    u = parent[v]
                    idx = parent_edge[v]
                    flow_adj[u][idx][1] -= bottleneck
                    rev_idx = flow_adj[u][idx][2]
                    flow_adj[v][rev_idx][1] += bottleneck
                    v = u

                total_flow += bottleneck

            return total_flow

        flow = max_flow()
        if flow == 0:
            return []

        # Reconstruct cycles from flow
        # An edge u->v in the flow means vertex u's out connects to vertex v's in
        # So we have a permutation: each vertex i has exactly one outgoing to some vertex
        # and one incoming from some vertex.

        # For each vertex i, find where its "out" (i+n) sends flow
        # i.e., find j such that flow_adj[i+n] has an edge to j with positive flow
        # (original cap was 1, so flow is 1 if used)

        next_vertex = [-1] * n
        for i in range(n):
            out_node = i + n
            for (v, cap, _) in flow_adj[out_node]:
                # Check if this edge has flow: original cap was 1, now cap is 0 if flow=1
                # The reverse edge would have cap > 0 if flow was sent
                # Actually, let's check: if the forward edge cap is 0 and it was originally 1,
                # then flow was sent. But we also have edges with original cap 0.
                # Better: check the reverse edge's capacity.
                pass

        # Let me redo this. The flow_adj structure:
        # flow_adj[u] contains [v, cap, rev_index]
        # When we send flow on edge u->v, we decrease cap and increase reverse cap.
        # So to check if flow was sent on original edge u->v with cap 1:
        #   flow_adj[u] entry for v has cap 0, and the reverse entry flow_adj[v][rev] has cap 1.

        # Simpler: just track which edges in the original graph have flow.
        # For each remaining edge (u,v), check if flow went u_out -> v_in or v_out -> u_in.

        # Let's build the next vertex mapping differently.
        # For each vertex i, look at flow_adj[i+n] (the out node).
        # For each entry [v, cap, rev], if v < n (it's an in-node) and the reverse
        # edge flow_adj[v][rev] has cap > 0, then flow was sent i -> v.

        next_vertex = [-1] * n
        for i in range(n):
            out_node = i + n
            for idx, (v, cap, rev) in enumerate(flow_adj[out_node]):
                if v < n:  # v is an in-node
                    # Check reverse edge
                    rev_cap = flow_adj[v][rev][1]
                    if rev_cap > 0:
                        # Flow was sent from i to v
                        next_vertex[i] = v
                        break

        # Now find cycles in the permutation next_vertex
        cycles = []
        visited = [False] * n
        for i in range(n):
            if not visited[i] and next_vertex[i] != -1:
                cycle = []
                cur = i
                while not visited[cur]:
                    visited[cur] = True
                    cycle.append(cur)
                    cur = next_vertex[cur]
                if len(cycle) >= 3:
                    cycles.append(cycle)
                # If len < 3, it's a 2-cycle or self-loop, which we discard
                # (but in a simple graph, self-loops aren't possible, and 2-cycles
                # would mean two vertices connected to each other, which is just one edge)

        return cycles

    def find_longest_cycle(adj, n, remaining_edges):
        """Find the longest cycle using DFS."""
        best_len = 0
        best_cycle = []

        for start in range(n):
            if not adj[start]:
                continue

            visited = {start: 0}
            parent = {start: None}
            in_stack = set([start])

            def dfs(u):
                nonlocal best_len, best_cycle
                for v in sorted(adj[u]):
                    e = (min(u, v), max(u, v))
                    if e not in remaining_edges:
                        continue
                    if v not in visited:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        in_stack.add(v)
                        dfs(v)
                        in_stack.remove(v)
                    elif v in in_stack:
                        length = visited[u] - visited[v] + 1
                        if length >= 3 and length > best_len:
                            cycle = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                cycle.append(cur)
                            best_cycle = cycle
                            best_len = length

            dfs(start)

            if time.time() > deadline:
                break

        return best_cycle

    def find_path_bfs(start, end, adj, remaining_edges, exclude_edge=None):
        """Find a path from start to end using remaining edges via BFS."""
        if start == end:
            return None
        queue = deque([(start, [start])])
        visited = {start}

        while queue and time.time() < deadline:
            node, path = queue.popleft()
            for neighbor in sorted(adj[node]):
                e = (min(node, neighbor), max(node, neighbor))
                if e not in remaining_edges:
                    continue
                if e == exclude_edge:
                    continue
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def try_merge_edges_into_cycles(pieces, n, adj, remaining_edges):
        """Try to merge single edges into cycles by finding paths between their endpoints."""
        improved = False
        i = 0
        while i < len(pieces) and time.time() < deadline:
            if len(pieces[i]) == 2:
                u, v = pieces[i]
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    path = find_path_bfs(u, v, adj, remaining_edges, exclude_edge=e)
                    if path and len(path) >= 3:
                        # Replace single edge with cycle
                        pieces[i] = path
                        for j in range(len(path)):
                            a, b = path[j], path[(j + 1) % len(path)]
                            ee = (min(a, b), max(a, b))
                            if ee in remaining_edges:
                                remaining_edges.discard(ee)
                                adj[a].discard(b)
                                adj[b].discard(a)
                        improved = True
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        return improved

    def try_merge_two_edges(pieces, n, adj, remaining_edges):
        """Try to merge pairs of single edges into a 4-cycle or longer cycle."""
        improved = False
        i = 0
        while i < len(pieces) and time.time() < deadline:
            if len(pieces[i]) == 2:
                u1, v1 = pieces[i]
                e1 = (min(u1, v1), max(u1, v1))
                if e1 in remaining_edges:
                    j = i + 1
                    while j < len(pieces) and time.time() < deadline:
                        if len(pieces[j]) == 2:
                            u2, v2 = pieces[j]
                            e2 = (min(u2, v2), max(u2, v2))
                            if e2 in remaining_edges and e2 != e1:
                                # Try to find a path from v1 to u2 and from v2 to u1
                                # that together form a cycle containing both edges
                                # Simpler: try to find a path from v1 to u2 using remaining edges
                                # (excluding e1 and e2), then check if v2 connects back
                                path1 = find_path_bfs(v1, u2, adj, remaining_edges, exclude_edge=e1)
                                if path1 and len(path1) >= 2:
                                    # path1 goes from v1 to u2
                                    # Now we need a path from v2 to u1 that doesn't use e1 or e2
                                    # and doesn't overlap with path1
                                    path2 = find_path_bfs(v2, u1, adj, remaining_edges, exclude_edge=e2)
                                    if path2 and len(path2) >= 2:
                                        # Check for vertex overlap (excluding endpoints)
                                        internal1 = set(path1[1:-1])
                                        internal2 = set(path2[1:-1])
                                        if not (internal1 & internal2):
                                            # Form cycle: u1 - v1 - path1 - u2 - v2 - path2 - u1
                                            # Wait, the edges are u1-v1 and u2-v2
                                            # Cycle: u1, v1, (path1 from v1 to u2), u2, v2, (path2 from v2 to u1)
                                            # But path1 starts at v1 and ends at u2
                                            # path2 starts at v2 and ends at u1
                                            # So cycle: u1, v1, path1[1:], u2, v2, path2[1:]
                                            # That is: u1, v1, path1[1], ..., u2, v2, path2[1], ..., u1
                                            cycle = [u1, v1] + path1[1:] + [v2] + path2[1:]
                                            # Check all vertices are distinct
                                            if len(set(cycle)) == len(cycle) and len(cycle) >= 4:
                                                # Verify all edges exist
                                                valid = True
                                                for k in range(len(cycle)):
                                                    a, b = cycle[k], cycle[(k + 1) % len(cycle)]
                                                    ee = (min(a, b), max(a, b))
                                                    if ee not in remaining_edges:
                                                        valid = False
                                                        break
                                                if valid:
                                                    # Remove both single edges and replace with cycle
                                                    pieces[i] = cycle
                                                    del pieces[j]
                                                    # Remove all edges of the cycle from remaining
                                                    for k in range(len(cycle)):
                                                        a, b = cycle[k], cycle[(k + 1) % len(cycle)]
                                                        ee = (min(a, b), max(a, b))
                                                        if ee in remaining_edges:
                                                            remaining_edges.discard(ee)
                                                            adj[a].discard(b)
                                                            adj[b].discard(a)
                                                    improved = True
                                                    break
                        j += 1
                    if improved:
                        continue
                i += 1
            else:
                i += 1
        return improved

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    for spec in specs:
        if time.time() > deadline:
            # Fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n = spec[0]
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []
        remaining_edges = set(edges)

        # Phase 1: Extract 2-factors greedily
        while remaining_edges and time.time() < deadline:
            cycles = extract_2factor(adj, n, remaining_edges)
            if cycles:
                for cycle in cycles:
                    pieces.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            remaining_edges.discard(e)
                            adj[u].discard(v)
                            adj[v].discard(u)
            else:
                # No more 2-factors; try longest cycle
                best_cycle = find_longest_cycle(adj, n, remaining_edges)
                if best_cycle:
                    pieces.append(best_cycle)
                    for i in range(len(best_cycle)):
                        u = best_cycle[i]
                        v = best_cycle[(i + 1) % len(best_cycle)]
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            remaining_edges.discard(e)
                            adj[u].discard(v)
                            adj[v].discard(u)
                else:
                    # No cycles left; add remaining edges as single pieces
                    for e in remaining_edges:
                        pieces.append(list(e))
                    break

        # Phase 2: Post-processing - merge single edges into cycles
        if time.time() < deadline:
            try_merge_edges_into_cycles(pieces, n, adj, remaining_edges)

        # Phase 3: Try to merge pairs of single edges
        if time.time() < deadline:
            try_merge_two_edges(pieces, n, adj, remaining_edges)

        results.append(pieces)

    return results
