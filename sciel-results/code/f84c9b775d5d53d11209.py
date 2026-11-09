import numpy as np
import time
import random

def construct():
    n = 20
    colors = 20
    
    def is_valid_add(adj, col, u, v, c):
        for w in range(n):
            if adj[v, w] == 1:
                cw = col[v, w]
                if cw == c:
                    return False
                for x in range(n):
                    if adj[w, x] == 1 and adj[x, u] == 1:
                        cxw = col[w, x]
                        cxu = col[x, u]
                        if len({c, cw, cxw, cxu}) < 4:
                            return False
        return True

    def is_valid_recolor(adj, col, u, v, new_c):
        # Check all 4-cycles containing (u,v)
        for w in range(n):
            if adj[v, w] == 1:
                cw = col[v, w]
                if cw == new_c:
                    return False
                for x in range(n):
                    if adj[w, x] == 1 and adj[x, u] == 1:
                        cxw = col[w, x]
                        cxu = col[x, u]
                        if len({new_c, cw, cxw, cxu}) < 4:
                            return False
        return True

    def count_edges(adj):
        return sum(adj) // 2

    def run_search(seed):
        random.seed(seed)
        adj = np.zeros((n, n), dtype=int)
        col = np.full((n, n), -1, dtype=int)
        
        # Init: Random tree
        # 0 connected to 1, 1 to 2, etc.
        for i in range(1, n):
            u, v = i-1, i
            c = i % colors
            adj[u, v] = 1
            adj[v, u] = 1
            col[u, v] = c
            col[v, u] = c
            
        best_count = count_edges(adj)
        
        start_time = time.time()
        deadline = start_time + 110
        
        while time.time() < deadline:
            # Try adding edges
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 0]
            random.shuffle(non_edges)
            added = False
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if adj[u, v] == 1:
                    continue
                for c in range(colors):
                    if is_valid_add(adj, col, u, v, c):
                        adj[u, v] = 1
                        adj[v, u] = 1
                        col[u, v] = c
                        col[v, u] = c
                        added = True
                        break
            if not added:
                # Try recoloring to unlock
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
                if edges:
                    u, v = random.choice(edges)
                    old_c = col[u, v]
                    new_c = random.randint(0, colors - 1)
                    if new_c != old_c and is_valid_recolor(adj, col, u, v, new_c):
                        col[u, v] = new_c
                        col[v, u] = new_c
                        # Don't count as progress, but might help next iter
                else:
                    break
                    
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                
        # Return best found? 
        # We should track the best state.
        # For simplicity, we return the current state if it's good, 
        # but we should store the best state seen.
        
        # Let's modify to store best state.
        
    # Better structure:
    
    def search_once(seed):
        random.seed(seed)
        adj = np.zeros((n, n), dtype=int)
        col = np.full((n, n), -1, dtype=int)
        
        for i in range(1, n):
            u, v = i-1, i
            c = i % colors
            adj[u, v] = 1
            adj[v, u] = 1
            col[u, v] = c
            col[v, u] = c
            
        best_adj = adj.copy()
        best_col = col.copy()
        best_count = count_edges(adj)
        
        start_time = time.time()
        # Allocate time based on remaining budget
        # We will call this multiple times? No, just one long run.
        
        # We need to track the best state found during the search.
        
        while time.time() < time.time() + 100: # Placeholder
            pass
            
    # Let's just do one long search and track best.
    
    adj = np.zeros((n, n), dtype=int)
    col = np.full((n, n), -1, dtype=int)
    for i in range(1, n):
        u, v = i-1, i
        c = i % colors
        adj[u, v] = 1
        adj[v, u] = 1
        col[u, v] = c
        col[v, u] = c
        
    best_adj = adj.copy()
    best_col = col.copy()
    best_count = count_edges(adj)
    
    start_time = time.time()
    deadline = start_time + 110
    
    while time.time() < deadline:
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 0]
        random.shuffle(non_edges)
        improved = False
        for u, v in non_edges:
            if time.time() > deadline:
                break
            if adj[u, v] == 1:
                continue
            for c in range(colors):
                if is_valid_add(adj, col, u, v, c):
                    adj[u, v] = 1
                    adj[v, u] = 1
                    col[u, v] = c
                    col[v, u] = c
                    improved = True
                    break
        if not improved:
            # Recolor
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
            if edges:
                u, v = random.choice(edges)
                old_c = col[u, v]
                new_c = random.randint(0, colors - 1)
                if new_c != old_c and is_valid_recolor(adj, col, u, v, new_c):
                    col[u, v] = new_c
                    col[v, u] = new_c
            else:
                break
                
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj.copy()
            best_col = col.copy()
            
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                result.append([i, j, best_col[i, j]])
                
    return result
