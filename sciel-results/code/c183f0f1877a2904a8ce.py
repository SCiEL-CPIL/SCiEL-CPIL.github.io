import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin before 150s kill
    
    def check_time():
        return time.time() - start_time > deadline
    
    # Helper: check if graph is triangle-free
    def is_triangle_free(adj, n):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[u]:
                        if w > v and w in adj[v]:
                            return False
        return True
    
    # Helper: find maximum independent set size (greedy + local search)
    def max_independent_set_size(adj, n, trials=50):
        best = 0
        for _ in range(trials):
            order = list(range(n))
            random.shuffle(order)
            current_ind = set()
            for v in order:
                if all(v not in adj[u] for u in current_ind):
                    current_ind.add(v)
            if len(current_ind) > best:
                best = len(current_ind)
                if best >= 10:
                    return best
        return best
    
    # Helper: construct Paley graph P_q
    def paley_graph(q):
        n = q
        qr_set = set()
        for i in range(1, q):
            qr_set.add((i * i) % q)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                diff = (u - v) % q
                if diff in qr_set:
                    edges.append([u, v])
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return n, edges, adj
    
    # Known good seeds: (n, edges, adj)
    # Try several Paley graphs and other constructions
    
    best_result = None
    best_n = 0
    
    # Try Paley graphs with prime q ≡ 1 (mod 4)
    # P_13: alpha=4, P_17: alpha=5, P_29: alpha=6, P_37: alpha=6, P_41: alpha=7?
    # We need alpha < 10, so many work. But we also need n as large as possible.
    
    candidate_primes = [13, 17, 29, 37, 41]
    
    for q in candidate_primes:
        if check_time():
            break
        
        n, edges, adj = paley_graph(q)
        
        # Verify triangle-free
        if not is_triangle_free(adj, n):
            continue
        
        # Check alpha < 10
        alpha = max_independent_set_size(adj, n, trials=100)
        if alpha >= 10:
            continue
        
        # This is a valid graph
        if n > best_n:
            best_n = n
            best_result = {'n': n, 'edges': edges}
    
    # If we found a valid graph, return it
    if best_result is not None:
        return best_result
    
    # Fallback: use a smaller known good graph
    # P_13 should always work
    n, edges, adj = paley_graph(13)
    if is_triangle_free(adj, n):
        alpha = max_independent_set_size(adj, n, trials=50)
        if alpha < 10:
            return {'n': n, 'edges': edges}
    
    # Ultimate fallback: empty graph on 9 vertices (no triangles, alpha=9 < 10)
    n = 9
    edges = []
    return {'n': n, 'edges': edges}
