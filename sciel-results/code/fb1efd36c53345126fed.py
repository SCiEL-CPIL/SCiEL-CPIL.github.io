    if best_cycle is None:
        # Fallback: Greedy cycle on full graph
        adj_full = [[] for _ in range(N)]
        for u, v in EDGE_ORDER:
            adj_full[u].append(v)
            adj_full[v].append(u)
            
        path = [0]
        visited = [False] * N
        visited[0] = True
        for _ in range(N - 1):
            u = path[-1]
            next_v = None
            for v in adj_full[u]:
                if not visited[v]:
                    next_v = v
                    break
            if next_v is None:
                # Should not happen if graph is connected
                next_v = 0 
            path.append(next_v)
            visited[next_v] = True
        # Check if it closes
        if not connected[path[-1]][0]:
             # If not, we need to find a valid cycle.
             # But for fallback, we just need *a* cycle.
             # The greedy might not close.
             # Let's just use the first N edges that form a path?
             # Actually, any Hamiltonian cycle in K_N works if edges exist.
             # Since EDGE_ORDER is a permutation of all edges, the full graph is K_N.
             # So a cycle 0-1-2-...-23-0 is valid IF those edges are in the set.
             # They are, because the set contains ALL edges.
             # Wait, EDGE_ORDER is a permutation of ALL edges.
             # So the full graph is K_N.
             # So 0-1-2-...-23-0 is a valid Hamiltonian cycle in the full graph.
             # Its score is max(pos(0,1), pos(1,2), ..., pos(23,0)).
             # This is a valid fallback.
             pass
        best_cycle = path
