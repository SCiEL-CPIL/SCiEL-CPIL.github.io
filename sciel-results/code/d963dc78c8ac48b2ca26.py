import time
import random

N = 20
C = 20
DEADLINE = 145.0

def _popcount(x):
    return bin(x).count('1')

def _can_add_edge(adj, u, v):
    # Check if adding (u,v) creates a C4
    # This happens if there are at least 2 common neighbors
    common = adj[u] & adj[v]
    return common.bit_count() <= 1

def _count_edges(adj):
    return sum(_popcount(adj[u]) for u in range(N)) // 2

def _greedy_fill(adj):
    # Modifies adj in place, returns count
    pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    random.shuffle(pairs)
    
    for u, v in pairs:
        if not (adj[u] >> v) & 1:
            if _can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    return _count_edges(adj)

def _local_search(adj, time_limit):
    best_adj = adj[:]
    best_count = _count_edges(adj)
    
    start_time = time.time()
    
    while time.time() < time_limit:
        # Try to add any available edge
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if not (adj[u] >> v) & 1:
                    if _can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        if improved:
            current_count = _count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
            continue
        
        # If stuck, perform a kick: remove a random edge and re-greedy
        edges = []
        for u in range(N):
            for v in range(u+1, N):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
        
        if not edges:
            break
            
        u, v = random.choice(edges)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Re-greedy
        _greedy_fill(adj)
        
        current_count = _count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
    return best_adj, best_count

def _color_graph(adj):
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (adj[u] >> v) & 1:
                edges.append((u, v))
                
    random.shuffle(edges)
    
    color_at = [[-1]*N for _ in range(N)]
    colored_edges = []
    
    for u, v in edges:
        used_colors = set()
        for w in range(N):
            if w != v and (adj[u] >> w) & 1:
                if color_at[u][w] != -1:
                    used_colors.add(color_at[u][w])
        for w in range(N):
            if w != u and (adj[v] >> w) & 1:
                if color_at[v][w] != -1:
                    used_colors.add(color_at[v][w])
                    
        for c in range(C):
            if c not in used_colors:
                color_at[u][v] = c
                color_at[v][u] = c
                colored_edges.append([u, v, c])
                break
                
    return colored_edges

def construct():
    start = time.time()
    best_adj = [0] * N
    best_len = 0
    
    # Run multiple restarts
    for seed in range(10000):
        if time.time() > DEADLINE - 5:
            break
            
        random.seed(seed)
        adj = [0] * N
        # Start with empty
        adj, count = _local_search(adj, DEADLINE - 10)
        
        if count > best_len:
            best_len = count
            best_adj = adj[:]
            
    if best_len == 0:
        return []
        
    colored = _color_graph(best_adj)
    return colored

if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
