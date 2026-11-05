import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 20
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(adj, u, v):
        nu = adj[u]
        nv = adj[v]
        
        # Check pairs in N(u)
        for x in range(n):
            if (nu >> x) & 1:
                nx = adj[x]
                for y in range(x + 1, n):
                    if (nu >> y) & 1:
                        ny = adj[y]
                        common = nx & ny & nv
                        if popcount(common) >= 2:
                            return False
                            
        # Check pairs in N(v)
        for x in range(n):
            if (nv >> x) & 1:
                nx = adj[x]
                for y in range(x + 1, n):
                    if (nv >> y) & 1:
                        ny = adj[y]
                        common = nx & ny & nu
                        if popcount(common) >= 2:
                            return False
                            
        return True

    def greedy_fill(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if can_add(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
        return adj

    def edge_count(adj):
        count = 0
        for i in range(n):
            count += popcount(adj[i])
        return count // 2

    def random_graph(p=0.5):
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    best_adj = [0] * n
    best_count = -1
    
    # Try multiple random starts
    for _ in range(50):
        if time.time() > deadline:
            break
        adj = random_graph(0.5)
        adj = greedy_fill(adj)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            
    # Local Search
    current_adj = best_adj[:]
    current_count = best_count
    
    while time.time() < deadline - 5.0:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (current_adj[u] >> v) & 1:
                    edges.append((u, v))
                    
        if not edges:
            break
            
        u, v = random.choice(edges)
        
        prev_adj = current_adj[:]
        prev_count = current_count
        
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        
        current_adj = greedy_fill(current_adj)
        new_count = edge_count(current_adj)
        
        if new_count > best_count:
            best_count = new_count
            best_adj = current_adj[:]
            
        if new_count > current_count:
            current_count = new_count
        else:
            current_adj = prev_adj
            current_count = prev_count
            
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])
                
    return result
