import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 15
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Start with K_{7,8}
    part_a = list(range(7))
    part_b = list(range(7, 15))
    
    adj = [set() for _ in range(n)]
    for u in part_a:
        for v in part_b:
            adj[u].add(v)
            adj[v].add(u)
            
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]
    
    def is_valid(adj):
        deg = get_degrees(adj)
        for u in range(n):
            for v in adj[u]:
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
        
    def edge_count(adj):
        return sum(len(adj[i]) for i in range(n)) // 2
        
    best_adj = [set(s) for s in adj]
    best_count = edge_count(best_adj)
    
    # Greedy Add
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u, v in all_pairs:
            if v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                if is_valid(adj):
                    improved = True
                    ec = edge_count(adj)
                    if ec > best_count:
                        best_count = ec
                        best_adj = [set(s) for s in adj]
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    
    # Convert to list
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
                
    return edges
