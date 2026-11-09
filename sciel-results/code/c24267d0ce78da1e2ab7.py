import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 20
    best_edges = []
    best_count = 0
    
    def count_edges(adj):
        return np.sum(adj) // 2

    def is_valid_add(adj, common, u, v):
        if adj[u][v]:
            return False
        if common[u][v] > 2:
            return False
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                if common[u][w] > 1:
                    return False
            if adj[u][w]:
                if common[v][w] > 1:
                    return False
        return True

    def add_edge(adj, common, u, v):
        adj[u][v] = 1
        adj[v][u] = 1
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] += 1
                common[w][u] += 1
            if adj[u][w]:
                common[v][w] += 1
                common[w][v] += 1
                
    def remove_edge(adj, common, u, v):
        adj[u][v] = 0
        adj[v][u] = 0
        
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:
                common[u][w] -= 1
                common[w][u] -= 1
            if adj[u][w]:
                common[v][w] -= 1
                common[w][v] -= 1

    def greedy_fill(adj, common, max_time=None):
        changed = True
        while changed:
            if max_time and time.time() > max_time:
                return
            changed = False
            edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
            random.shuffle(edges_to_try)
            for u, v in edges_to_try:
                if is_valid_add(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    changed = True
                    if max_time and time.time() > max_time:
                        return

    def perturb_and_fill(adj, common, max_time=None):
        # Remove a random edge
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
        if not edges:
            return
        u, v = random.choice(edges)
        remove_edge(adj, common, u, v)
        
        # Try to add it back or other edges
        greedy_fill(adj, common, max_time)

    # Multiple restarts
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        # Initialize
        # Start with empty or random
        # Random with low density to allow growth
        p = 0.2 + 0.3 * random.random()
        adj = np.zeros((n, n), dtype=int)
        common = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    common[i][j] = np.dot(adj[i], adj[j])
        
        # Greedy fill
        greedy_fill(adj, common, max_time=deadline)
        
        # Perturb and fill a few times
        for _ in range(5):
            if time.time() > deadline:
                break
            perturb_and_fill(adj, common, max_time=deadline)
            
        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        best_edges.append([i, j])

    return best_edges
