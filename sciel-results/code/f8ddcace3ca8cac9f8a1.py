import random
import time

def construct():
    n = 20
    deadline = time.time() + 110
    
    def is_valid_add(adj, u, v):
        if adj[u][v]:
            return False
        common = []
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                common.append(w)
        if len(common) < 3:
            return True
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                for k in range(j+1, len(common)):
                    s = (common[i], common[j], common[k])
                    for x in range(n):
                        if x == u or x == v or x in s:
                            continue
                        if adj[x][s[0]] and adj[x][s[1]] and adj[x][s[2]]:
                            if not adj[x][u] and not adj[x][v]:
                                return False
        return True

    def count_edges(adj):
        c = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    c += 1
        return c

    def try_add_edge(adj, u, v):
        if is_valid_add(adj, u, v):
            adj[u][v] = 1
            adj[v][u] = 1
            return True
        return False

    def add_max_edges(adj):
        # Greedily add as many edges as possible
        added = 0
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not adj[u][v]:
                        if try_add_edge(adj, u, v):
                            improved = True
                            added += 1
        return added

    best_adj = None
    best_count = 0
    
    for seed in range(50):
        if time.time() > deadline:
            break
        random.seed(seed)
        adj = [[0]*n for _ in range(n)]
        
        # Greedy initialization
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            try_add_edge(adj, u, v)
            
        current_count = count_edges(adj)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add
            added = add_max_edges(adj)
            if added > 0:
                improved = True
                current_count += added
                continue
                
            # If stuck, try removing one edge and re-filling
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
            if not existing:
                break
                
            # Try a few random removals
            best_neighbor_count = current_count
            best_neighbor_adj = None
            
            # Sample up to 20 random edges to remove
            for _ in range(20):
                if time.time() > deadline:
                    break
                u, v = random.choice(existing)
                # Save state
                saved_adj = [row[:] for row in adj]
                
                # Remove
                adj[u][v] = 0
                adj[v][u] = 0
                
                # Re-fill
                add_max_edges(adj)
                new_count = count_edges(adj)
                
                if new_count > best_neighbor_count:
                    best_neighbor_count = new_count
                    best_neighbor_adj = [row[:] for row in adj]
                    
                # Restore
                adj = [row[:] for row in saved_adj]
                
            if best_neighbor_adj:
                adj = best_neighbor_adj
                current_count = best_neighbor_count
                improved = True
            else:
                break
                
        if current_count > best_count:
            best_count = current_count
            best_adj = [row[:] for row in adj]
            
    if best_adj is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([i, j])
    return result
