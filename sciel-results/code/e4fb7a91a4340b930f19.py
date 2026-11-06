import numpy as np
import time
import random
from itertools import combinations

def has_c5(adj):
    n = adj.shape[0]
    # Check all 5-subsets
    for combo in combinations(range(n), 5):
        u, v, w, x, y = combo
        # Check 12 cycles
        # 0-1-2-3-4-0
        if adj[u][v] and adj[v][w] and adj[w][x] and adj[x][y] and adj[y][u]:
            return True
        # 0-1-2-4-3-0
        if adj[u][v] and adj[v][w] and adj[w][y] and adj[y][x] and adj[x][u]:
            return True
        # 0-1-3-2-4-0
        if adj[u][v] and adj[v][x] and adj[x][w] and adj[w][y] and adj[y][u]:
            return True
        # 0-1-3-4-2-0
        if adj[u][v] and adj[v][x] and adj[x][y] and adj[y][w] and adj[w][u]:
            return True
        # 0-1-4-2-3-0
        if adj[u][v] and adj[v][y] and adj[y][w] and adj[w][x] and adj[x][u]:
            return True
        # 0-1-4-3-2-0
        if adj[u][v] and adj[v][y] and adj[y][x] and adj[x][w] and adj[w][u]:
            return True
        # 0-2-1-3-4-0
        if adj[u][w] and adj[w][v] and adj[v][x] and adj[x][y] and adj[y][u]:
            return True
        # 0-2-1-4-3-0
        if adj[u][w] and adj[w][v] and adj[v][y] and adj[y][x] and adj[x][u]:
            return True
        # 0-2-3-1-4-0
        if adj[u][w] and adj[w][x] and adj[x][v] and adj[v][y] and adj[y][u]:
            return True
        # 0-2-3-4-1-0
        if adj[u][w] and adj[w][x] and adj[x][y] and adj[y][v] and adj[v][u]:
            return True
        # 0-2-4-1-3-0
        if adj[u][w] and adj[w][y] and adj[y][v] and adj[v][x] and adj[x][u]:
            return True
        # 0-2-4-3-1-0
        if adj[u][w] and adj[w][y] and adj[y][x] and adj[x][v] and adj[v][u]:
            return True
    return False

def greedy_independent_set_size(adj):
    n = adj.shape[0]
    remaining = list(range(n))
    size = 0
    while remaining:
        min_deg = float('inf')
        min_idx = -1
        for i in remaining:
            deg = sum(adj[i][j] for j in remaining)
            if deg < min_deg:
                min_deg = deg
                min_idx = i
        if min_idx == -1:
            break
        size += 1
        to_remove = set()
        to_remove.add(min_idx)
        for j in remaining:
            if adj[min_idx][j]:
                to_remove.add(j)
        remaining = [v for v in remaining if v not in to_remove]
    return size

def exact_max_independent_set(adj, lower_bound=0):
    n = adj.shape[0]
    best = lower_bound
    
    # Sort vertices by degree (descending) to prune better?
    # Or just use natural order.
    
    def dfs(idx, current_size, forbidden):
        nonlocal best
        if current_size > best:
            best = current_size
            if best >= 9:
                return
        if idx == n:
            return
        
        # Pruning
        if current_size + (n - idx) <= best:
            return
        
        # Try adding vertex idx
        if not forbidden[idx]:
            new_forbidden = forbidden.copy()
            new_forbidden[idx] = True
            for j in range(idx + 1, n):
                if adj[idx][j]:
                    new_forbidden[j] = True
            dfs(idx + 1, current_size + 1, new_forbidden)
            if best >= 9:
                return
        
        # Try not adding vertex idx
        dfs(idx + 1, current_size, forbidden)
        
    forbidden = [False] * n
    dfs(0, 0, forbidden)
    return best

def is_valid_graph(adj):
    # Check C5
    if has_c5(adj):
        return False
    
    # Check Alpha
    # Greedy first
    g_size = greedy_independent_set_size(adj)
    if g_size >= 9:
        return False
    
    # Exact check
    e_size = exact_max_independent_set(adj, g_size)
    if e_size >= 9:
        return False
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 28
    best_edges = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]
    
    for n in range(35, 28, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        attempts = 0
        
        while not found and time.time() < deadline - 10:
            attempts += 1
            
            # Generate random graph
            # Try different p values
            p = random.uniform(0.4, 0.8)
            adj = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
                        
            if is_valid_graph(adj):
                found = True
                break
            
            # Local Search
            # We try to flip edges to satisfy constraints
            for step in range(200):
                if time.time() > deadline - 10:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j: continue
                if i > j: i, j = j, i
                
                adj[i][j] ^= 1
                adj[j][i] ^= 1
                
                if is_valid_graph(adj):
                    found = True
                    break
                else:
                    # Revert with high probability to avoid wandering
                    if random.random() > 0.2:
                        adj[i][j] ^= 1
                        adj[j][i] ^= 1
                        
        if found:
            best_n = n
            best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j] == 1]
            
    return {'n': best_n, 'edges': best_edges}
