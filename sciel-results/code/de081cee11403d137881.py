def construct():
    n = 15
    import time
    import random
    deadline = time.time() + 82  # safety margin before 90s kill

    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)  # 105

    # Precompute: for each edge, which other edges form a 3-path with it
    # A 3-path s-a-b-t uses edges (s,a), (a,b), (b,t)
    # The middle edge is (a,b). For a given middle edge (a,b),
    # s is a neighbor of a (s != b), t is a neighbor of b (t != a, t != s)
    # The endpoints are s and t.
    
    # For each edge (a,b) as the middle edge of a 3-path:
    # For each s in adj[a] \ {b}, for each t in adj[b] \ {a, s}:
    #   constraint: deg(s) != deg(t)
    #   where deg(s) = sum of edge variables incident to s
    #   and deg(t) = sum of edge variables incident to t

    # Let's index edges 0..m-1
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # For each vertex, list of incident edge indices
    vertex_edges = [[] for _ in range(n)]
    for i, (u, v) in enumerate(all_edges):
        vertex_edges[u].append(i)
        vertex_edges[v].append(i)

    # For each middle edge (a,b), precompute the set of (s, t) endpoint pairs
    # and for each such pair, the set of edge indices incident to s and to t
    middle_edge_paths = {}  # middle_edge_idx -> list of (s, t, deg_s_edges, deg_t_edges)
    for mid_idx, (a, b) in enumerate(all_edges):
        paths = []
        for s in range(n):
            if s == a or s == b:
                continue
            if (s, a) not in edge_index:
                continue
            for t in range(n):
                if t == a or t == b or t == s:
                    continue
                if (t, b) not in edge_index:
                    continue
                # s-a-b-t is a valid 3-path
                # deg(s) = sum of x_e for e in vertex_edges[s]
                # deg(t) = sum of x_e for e in vertex_edges[t]
                # Constraint: deg(s) != deg(t)
                # i.e., NOT (deg(s) == deg(t))
                # i.e., deg(s) - deg(t) != 0
                # This is a linear constraint: sum_{e in vertex_edges[s]} x_e - sum_{e in vertex_edges[t]} x_e != 0
                # For Max-SAT: we want to maximize sum x_e subject to these != 0 constraints
                # We can encode != 0 as: at least one of (diff >= 1) or (diff <= -1)
                # But that's complex. Alternative: just check validity directly.
                paths.append((s, t))
        middle_edge_paths[mid_idx] = paths

    def check_validity(edge_set, adj, deg):
        """Check if no 3-edge path has endpoints of equal degree."""
        for (a, b) in edge_set:
            for s in adj[a]:
                if s == b:
                    continue
                for t in adj[b]:
                    if t == a or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def count_violations(edge_set, adj, deg):
        """Count number of violating 3-paths."""
        count = 0
        for (a, b) in edge_set:
            for s in adj[a]:
                if s == b:
                    continue
                for t in adj[b]:
                    if t == a or t == s:
                        continue
                    if deg[s] == deg[t]:
                        count += 1
        return count

    def build_adj_deg(edge_set):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def try_add_edge(edge_set, adj, deg, u, v):
        """Try adding edge (u,v). Return (success, new_adj, new_deg)."""
        if u == v:
            return False, adj, deg
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False, adj, deg

        new_deg = deg[:]
        new_deg[u] += 1
        new_deg[v] += 1

        # Check new paths through (u,v)
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if new_deg[s] == new_deg[t]:
                    return False, adj, deg

        # Check existing paths where endpoint degree changed (s or t is u or v)
        for (a, b) in edge_set:
            if a == u or a == v or b == u or b == v:
                # This edge is incident to u or v, so paths through it may have changed endpoints
                pass
            for s in adj[a]:
                if s == b:
                    continue
                for t in adj[b]:
                    if t == a or t == s:
                        continue
                    if s == u or t == u or s == v or t == v:
                        if new_deg[s] == new_deg[t]:
                            return False, adj, deg

        return True, adj, new_deg

    def greedy_add_edges(edge_set, adj, deg, time_limit):
        """Greedily add edges that maintain validity."""
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > time_limit:
                        break
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        continue
                    success, new_adj, new_deg = try_add_edge(edge_set, adj, deg, u, v)
                    if success:
                        edge_set.add(key)
                        adj[u].add(v)
                        adj[v].add(u)
                        deg = new_deg
                        changed = True
            if time.time() > time_limit:
                break
        return edge_set, adj, deg

    def local_search_remove(edge_set, adj, deg, time_limit):
        """Remove edges to eliminate violations, then try to re-add."""
        while time.time() < time_limit:
            # Count violations
            violations = count_violations(edge_set, adj, deg)
            if violations == 0:
                # Try to add more edges
                edge_set, adj, deg = greedy_add_edges(edge_set, adj, deg, min(time.time() + 1, time_limit))
                continue

            # Find the edge involved in the most violations
            # For each edge, count how many violating paths it's part of
            edge_violation_count = {e: 0 for e in edge_set}
            for (a, b) in edge_set:
                for s in adj[a]:
                    if s == b:
                        continue
                    for t in adj[b]:
                        if t == a or t == s:
                            continue
                        if deg[s] == deg[t]:
                            # This path s-a-b-t is violating
                            # It involves edges (s,a), (a,b), (b,t)
                            edge_violation_count[(min(s,a), max(s,a))] += 1
                            edge_violation_count[(min(a,b), max(a,b))] += 1
                            edge_violation_count[(min(b,t), max(b,t))] += 1

            # Remove the edge with the most violations
            worst_edge = max(edge_violation_count, key=lambda e: edge_violation_count[e])
            if edge_violation_count[worst_edge] == 0:
                break

            # Remove it
            u, v = worst_edge
            edge_set.remove(worst_edge)
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1

        # After removal, try to add edges back
        edge_set, adj, deg = greedy_add_edges(edge_set, adj, deg, min(time.time() + 2, time_limit))
        return edge_set, adj, deg

    def random_restart_search(time_limit):
        """Try random initial subsets and improve."""
        best_edges = None
        best_count = -1

        # Try several random starting points
        for attempt in range(20):
            if time.time() > time_limit:
                break

            # Start with a random subset of edges
            random.shuffle(all_edges)
            edge_set = set()
            adj = [set() for _ in range(n)]
            deg = [0] * n

            # Add random edges until we can't add more
            for (u, v) in all_edges:
                if time.time() > time_limit:
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    continue
                success, new_adj, new_deg = try_add_edge(edge_set, adj, deg, u, v)
                if success:
                    edge_set.add(key)
                    adj[u].add(v)
                    adj[v].add(u)
                    deg = new_deg

            # Now do local search
            edge_set, adj, deg = local_search_remove(edge_set, adj, deg, min(time.time() + 2, time_limit))

            if len(edge_set) > best_count:
                best_count = len(edge_set)
                best_edges = set(edge_set)

        return best_edges

    # Phase 1: Start with K_{7,8} and greedily add edges
    edge_set = set()
    for u in range(7):
        for v in range(7, 15):
            edge_set.add((u, v))

    adj, deg = build_adj_deg(edge_set)
    edge_set, adj, deg = greedy_add_edges(edge_set, adj, deg, time.time() + 10)

    best_edges = set(edge_set)
    best_count = len(edge_set)

    # Phase 2: Local search with removal
    edge_set, adj, deg = local_search_remove(edge_set, adj, deg, time.time() + 15)
    if len(edge_set) > best_count:
        best_count = len(edge_set)
        best_edges = set(edge_set)

    # Phase 3: Random restarts
    random.seed(42)
    restart_edges = random_restart_search(time.time() + 40)
    if restart_edges is not None and len(restart_edges) > best_count:
        best_count = len(restart_edges)
        best_edges = restart_edges

    # Phase 4: Final greedy pass on the best solution
    if time.time() < deadline - 5:
        adj, deg = build_adj_deg(best_edges)
        best_edges, adj, deg = greedy_add_edges(best_edges, adj, deg, min(time.time() + 5, deadline - 2))

    # Build final edge list
    final_edges = []
    for (u, v) in best_edges:
        final_edges.append([u, v])

    return final_edges
