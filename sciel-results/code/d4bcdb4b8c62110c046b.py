import time

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Helper: check if a graph is valid
    def is_valid(adj, deg):
        for s in range(n):
            for u in range(n):
                if u == s or not adj[s][u]:
                    continue
                for v in range(n):
                    if v == u or v == s or not adj[u][v]:
                        continue
                    for t in range(n):
                        if t == s or t == u or t == v or not adj[v][t]:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def get_degrees(adj):
        return [sum(adj[i]) for i in range(n)]
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    # Start with K_{7,8} as baseline
    part_a = list(range(7))
    part_b = list(range(7, 15))
    
    best_adj = [[False] * n for _ in range(n)]
    for u in part_a:
        for v in part_b:
            best_adj[u][v] = True
            best_adj[v][u] = True
    
    best_deg = get_degrees(best_adj)
    best_edges = count_edges(best_adj)
    
    # All possible edges
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Try to add edges one by one, checking validity
    # Use a greedy approach: try adding each missing edge, keep if valid
    for _ in range(200):  # Multiple passes
        if time.time() > deadline:
            break
        improved = False
        for (u, v) in all_pairs:
            if time.time() > deadline:
                break
            if best_adj[u][v]:
                continue
            # Try adding edge (u, v)
            best_adj[u][v] = True
            best_adj[v][u] = True
            new_deg = get_degrees(best_adj)
            if is_valid(best_adj, new_deg):
                best_deg = new_deg
                best_edges = count_edges(best_adj)
                improved = True
            else:
                # Revert
                best_adj[u][v] = False
                best_adj[v][u] = False
    
    # Also try removing edges to allow adding more others
    # Try a local search: for each pair of edges, try swapping
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i, (u1, v1) in enumerate(all_pairs):
            if time.time() > deadline:
                break
            if not best_adj[u1][v1]:
                continue
            for (u2, v2) in all_pairs[i+1:]:
                if time.time() > deadline:
                    break
                if best_adj[u2][v2]:
                    continue
                # Try swap: remove (u1,v1), add (u2,v2)
                best_adj[u1][v1] = False
                best_adj[v1][u1] = False
                best_adj[u2][v2] = True
                best_adj[v2][u2] = True
                new_deg = get_degrees(best_adj)
                new_edges = count_edges(best_adj)
                if new_edges > best_edges and is_valid(best_adj, new_deg):
                    best_deg = new_deg
                    best_edges = new_edges
                    improved = True
                else:
                    # Revert
                    best_adj[u1][v1] = True
                    best_adj[v1][u1] = True
                    best_adj[u2][v2] = False
                    best_adj[v2][u2] = False
        if not improved:
            break
    
    # Build result
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    
    return edges
