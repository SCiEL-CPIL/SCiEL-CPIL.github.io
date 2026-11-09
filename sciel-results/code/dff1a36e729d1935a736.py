import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    n = 16
    adj = [[0]*n for _ in range(n)]
    
    def add_edge(u, v):
        if u != v and not adj[u][v]:
            adj[u][v] = 1
            adj[v][u] = 1
            
    def remove_edge(u, v):
        if u != v and adj[u][v]:
            adj[u][v] = 0
            adj[v][u] = 0
            
    def count_edges():
        return sum(sum(row) for row in adj) // 2
        
    # Q3 edges
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
    q3_edge_set = set(q3_edges) | set((v, u) for u, v in q3_edges)
    
    def has_q3_in_induced(verts):
        # verts: tuple of 8 global indices
        # Count edges
        e_count = 0
        local_adj = [[0]*8 for _ in range(8)]
        for i in range(8):
            for j in range(8):
                if i != j and adj[verts[i]][verts[j]]:
                    local_adj[i][j] = 1
                    e_count += 1
        e_count //= 2
        if e_count < 12:
            return False
            
        # Degree check
        for i in range(8):
            if sum(local_adj[i]) < 3:
                return False
                
        # Backtracking
        assigned = [-1] * 8
        assigned[0] = 0
        
        def search(next_q3):
            if next_q3 == 8:
                return True
            for local_v in range(8):
                if local_v in assigned:
                    continue
                ok = True
                for q3_u in range(next_q3):
                    if (q3_u, next_q3) in q3_edge_set:
                        if not local_adj[assigned[q3_u]][local_v]:
                            ok = False
                            break
                if not ok:
                    continue
                assigned[next_q3] = local_v
                if search(next_q3 + 1):
                    return True
                assigned[next_q3] = -1
            return False
            
        return search(1)
        
    def has_q3_global():
        for subset in itertools.combinations(range(n), 8):
            if has_q3_in_induced(subset):
                return True, subset
        return False, None
        
    best_edges = 0
    best_adj = None
    
    # Restart with different initializations
    initializers = [
        lambda: None, # K8,8
        lambda: None, # Random 0.8
        lambda: None, # Random 0.6
    ]
    
    for init_idx in range(3):
        if time.time() > deadline:
            break
            
        # Reset adj
        adj = [[0]*n for _ in range(n)]
        
        if init_idx == 0:
            for u in range(8):
                for v in range(8, 16):
                    add_edge(u, v)
        else:
            p = 0.8 if init_idx == 1 else 0.6
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        add_edge(u, v)
                        
        # Phase 1: Remove
        while time.time() < deadline:
            found, subset = has_q3_global()
            if not found:
                break
            edges_in_subset = []
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[subset[i]][subset[j]]:
                        edges_in_subset.append((subset[i], subset[j]))
            if not edges_in_subset:
                break
            u, v = random.choice(edges_in_subset)
            remove_edge(u, v)
            
        # Phase 2: Add
        while time.time() < deadline:
            added = False
            missing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if not adj[u][v]]
            random.shuffle(missing_edges)
            for u, v in missing_edges:
                if time.time() > deadline:
                    break
                add_edge(u, v)
                found, _ = has_q3_global()
                if not found:
                    added = True
                    break
                else:
                    remove_edge(u, v)
            if not added:
                break
                
        if count_edges() > best_edges:
            best_edges = count_edges()
            best_adj = [row[:] for row in adj]
            
    # Fallback to seed if no better found
    if best_adj is None:
        edges = [[a, b] for a in range(3) for b in range(3, 16)]
        return edges
        
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                edges.append([u, v])
    return edges
