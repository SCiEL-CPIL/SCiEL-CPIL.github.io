import random
import time

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n = 40
    m = 401
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set_all = set(all_edges)
    
    def max_book_from_adj(adj, edges):
        mx = 0
        for u, v in edges:
            b = len(adj[u] & adj[v])
            if b > mx:
                mx = b
        return mx
    
    rng = random.Random(7)
    best_edges = None
    best_score = float('inf')
    
    for restart in range(50):
        if time.time() > deadline - 30:
            break
        
        edges = set(rng.sample(all_edges, m))
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        score = max_book_from_adj(adj, edges)
        
        if score < best_score:
            best_score = score
            best_edges = set(edges)
        
        improved = True
        iters = 0
        while improved and time.time() < deadline - 20:
            improved = False
            iters += 1
            if iters > 10000:
                break
            
            edge_list = list(edges)
            rng.shuffle(edge_list)
            
            for (u, v) in edge_list:
                if time.time() > deadline - 20:
                    break
                
                cur_book = len(adj[u] & adj[v])
                if cur_book < score - 1:
                    continue
                
                non_edges = edge_set_all - edges
                if not non_edges:
                    break
                
                ne_list = list(non_edges)
                rng.shuffle(ne_list)
                for new_edge in ne_list[:30]:
                    if time.time() > deadline - 20:
                        break
                    a, b = new_edge
                    if (a, b) == (u, v):
                        continue
                    
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[a].add(b)
                    adj[b].add(a)
                    
                    trial_score = max_book_from_adj(adj, edges - {(u,v)} | {new_edge})
                    
                    if trial_score < score:
                        score = trial_score
                        edges = edges - {(u,v)} | {new_edge}
                        improved = True
                        break
                    else:
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[a].discard(b)
                        adj[b].discard(a)
                
                if improved:
                    break
        
        if score < best_score:
            best_score = score
            best_edges = set(edges)
        
        if best_score <= 8:
            break
    
    return [[u, v] for (u, v) in best_edges]
