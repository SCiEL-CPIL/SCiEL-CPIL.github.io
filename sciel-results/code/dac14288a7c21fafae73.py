import time
import random

def try_construct(n, target_degree):
    neighbors = [set() for _ in range(n)]
    edges = []
    
    while True:
        min_deg = min(len(neighbors[u]) for u in range(n))
        if min_deg >= target_degree:
            edge_list = []
            for u in range(n):
                for v in neighbors[u]:
                    if u < v:
                        edge_list.append([u, v])
            return {'n': n, 'edges': edge_list}
        
        candidates_u = [u for u in range(n) if len(neighbors[u]) == min_deg]
        u = random.choice(candidates_u)
        
        valid_vs = []
        for v in range(n):
            if v == u or v in neighbors[u]:
                continue
            if not (neighbors[u] & neighbors[v]):
                valid_vs.append(v)
                
        if not valid_vs:
            return None
            
        valid_vs.sort(key=lambda x: len(neighbors[x]))
        k = min(10, len(valid_vs))
        v = random.choice(valid_vs[:k])
        
        neighbors[u].add(v)
        neighbors[v].add(u)
        edges.append([u, v])

def construct():
    start_time = time.time()
    total_deadline = start_time + 100  # 100s limit
    
    # Allocate time to each n
    # 47: 40s, 46: 40s, 45: 10s, 44: 5s, 43: 2.5s, 42: 1s, 41: 0.5s, 40: 0.5s
    time_alloc = {
        47: 40,
        46: 40,
        45: 10,
        44: 5,
        43: 2,
        42: 1,
        41: 0.5,
        40: 0.5
    }
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > total_deadline:
            break
            
        target_degree = n - 39
        alloc = time_alloc.get(n, 1)
        n_deadline = time.time() + alloc
        
        while time.time() < n_deadline:
            res = try_construct(n, target_degree)
            if res is not None:
                return res
                
    # Fallback
    return {'n': 39, 'edges': []}
