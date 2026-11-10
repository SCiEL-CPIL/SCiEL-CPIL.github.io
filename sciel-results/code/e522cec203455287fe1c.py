import time
import random
from itertools import combinations

def count_distinct_cycle_lengths(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    for i in range(n):
        if len(adj[i]) < 3:
            return None
        
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w > v and w != u and (min(u, w), max(u, w)) in edge_set:
                    return None
    
    cycle_lengths = set()
    
    def dfs(node, start, path_len, visited):
        for neighbor in adj[node]:
            if neighbor == start and path_len >= 3:
                cycle_lengths.add(path_len)
            elif not visited[neighbor] and neighbor > start:
                visited[neighbor] = True
                dfs(neighbor, start, path_len + 1, visited)
                visited[neighbor] = False
                
    for i in range(n):
        visited = [False] * n
        visited[i] = True
        dfs(i, i, 1, visited)
        
    if not cycle_lengths:
        return None
        
    return len(cycle_lengths)

def get_best_known():
    edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
    return edges

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_edges = get_best_known()
    best_value = 2
    
    n = 12
    all_edges = list(combinations(range(n), 2))
    
    random.seed(77777)
    
    for attempt in range(3000):
        if time.time() > deadline:
            break
            
        num_edges = random.randint(18, 26)
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        edges = [[u, v] for u, v in shuffled[:num_edges]]
        
        val = count_distinct_cycle_lengths(edges, n)
        if val is not None and val < best_value:
            best_value = val
            best_edges = edges
            if val == 1:
                break
    
    return [[int(u), int(v)] for u, v in best_edges]
