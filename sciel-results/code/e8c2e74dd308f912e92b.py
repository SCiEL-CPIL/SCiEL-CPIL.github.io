import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 15
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_pairs)
    
    def build_base():
        edges = set()
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u in range(7):
            for v in range(7, 15):
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        return edges, adj, deg
    
    def is_valid_full(adj, deg):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for s in adj[u]:
                        if s == v:
                            continue
                        ds = deg[s]
                        for t in adj[v]:
                            if t == u or t == s:
                                continue
                            if ds == deg[t]:
                                return False
                    for s in adj[v]:
                        if s == u:
                            continue
                        ds = deg[s]
                        for t in adj[u]:
                            if t == v or t == s:
                                continue
                            if ds == deg[t]:
                                return False
        return True
    
    def can_add(u, v, adj, deg):
        du_new = deg[u] + 1
        dv_new = deg[v] + 1
        
        # Case 1: s-u-v-t (new edge u-v is middle)
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
        
        # Case 2: u-v-x-y (new edge u-v is first, u is endpoint)
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if deg[y] == du_new:
                    return False
        
        # Case 3: x-y-v-u (new edge u-v is last, u is endpoint)
        for x in adj[u]:
            if x == v:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
            if deg[x] == du_new:
                return False
        
        # Case 4: s-v-u-t (new edge u-v is middle, reversed)
        for s in adj[v]:
            if s == u:
                continue
            ds = deg[s]
            for t in adj[u]:
                if t == v or t == s:
                    continue
                if ds == deg[t]:
                    return False
        
        # Case 5: v-u-x-y (new edge u-v is first, v is endpoint)
        for x in adj[u]:
            if x == v:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
                if deg[y] == dv_new:
                    return False
        
        # Case 6: x-y-u-v (new edge u-v is last, v is endpoint)
        for x in adj[v]:
            if x == u:
                continue
            for y in adj[x]:
                if y == u or y == v:
                    continue
            if deg[x] == dv_new:
                return False
        
        return True
    
    def add_edge_state(u, v, adj, deg):
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    
    def remove_edge_state(u, v, adj, deg):
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1
    
    # Build base
    base_edges, adj, deg = build_base()
    assert is_valid_full(adj, deg)
    
    # Greedy phase: try adding intra-partition edges
    intra_candidates = [(i, j) for i in range(7) for j in range(i+1, 7)] + \
                       [(i, j) for i in range(7, 15) for j in range(i+1, 15)]
    
    current_edges = set(base_edges)
    
    # Try all intra candidates greedily
    for u, v in intra_candidates:
        if (u, v) not in current_edges and can_add(u, v, adj, deg):
            add_edge_state(u, v, adj, deg)
            current_edges.add((u, v))
    
    best_edges = set(current_edges)
    best_count = len(best_edges)
    
    # Simulated annealing phase
    random.seed(42)
    T = 500.0
    T_min = 0.01
    cooling = 0.99995
    
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Choose move
        r = random.random()
        
        if r < 0.25:
            # Add an edge
            non_edges = [p for p in all_pairs if p not in current_edges]
            if non_edges:
                u, v = random.choice(non_edges)
                if can_add(u, v, adj, deg):
                    add_edge_state(u, v, adj, deg)
                    current_edges.add((u, v))
                    no_improve = 0
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
                else:
                    no_improve += 1
        elif r < 0.5:
            # Remove an edge
            if current_edges:
                u, v = random.choice(tuple(current_edges))
                remove_edge_state(u, v, adj, deg)
                current_edges.discard((u, v))
                no_improve += 1
        else:
            # Swap
            if current_edges and len(current_edges) < m:
                u1, v1 = random.choice(tuple(current_edges))
                non_edges = [p for p in all_pairs if p not in current_edges]
                if non_edges:
                    u2, v2 = random.choice(non_edges)
                    remove_edge_state(u1, v1, adj, deg)
                    current_edges.discard((u1, v1))
                    
                    if can_add(u2, v2, adj, deg):
                        add_edge_state(u2, v2, adj, deg)
                        current_edges.add((u2, v2))
                        no_improve = 0
                        if len(current_edges) > best_count:
                            best_count = len(current_edges)
                            best_edges = set(current_edges)
                    else:
                        add_edge_state(u1, v1, adj, deg)
                        current_edges.add((u1, v1))
                        no_improve += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final verification
    adj_final = [set() for _ in range(n)]
    deg_final = [0] * n
    for u, v in best_edges:
        adj_final[u].add(v)
        adj_final[v].add(u)
        deg_final[u] += 1
        deg_final[v] += 1
    
    if not is_valid_full(adj_final, deg_final):
        best_edges = set(base_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result
