import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n = 40
    target_edges = 401
    
    # Initialize random graph
    random.seed(42)
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible_edges)
    initial_edges = all_possible_edges[:target_edges]
    
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in initial_edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        
    # Calculate initial books
    books = {}
    max_book = 0
    for u, v in edge_set:
        common = len(adj[u] & adj[v])
        books[(u, v)] = common
        books[(v, u)] = common
        if common > max_book:
            max_book = common
            
    best_max = max_book
    best_edge_set = set(edge_set)
    
    # Simulated Annealing
    T = 5.0
    cooling = 0.9995
    
    # Helper to compute global max
    def get_global_max():
        mx = 0
        for u, v in edge_set:
            b = books.get((u,v), 0)
            if b > mx:
                mx = b
        return mx

    while time.time() - start_time < DEADLINE:
        if T < 0.01:
            T = 0.01
            
        # Pick an edge to remove
        u, v = random.choice(list(edge_set))
        
        # Pick a non-edge to add
        added = False
        for _ in range(30):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b:
                continue
            ka, kb = min(a,b), max(a,b)
            if (ka, kb) in edge_set:
                continue
                
            # Apply move
            adj[u].discard(v)
            adj[v].discard(u)
            edge_set.discard((u, v))
            
            adj[a].add(b)
            adj[b].add(a)
            edge_set.add((ka, kb))
            
            # Update books for affected
            affected = set()
            for x in [u, v, a, b]:
                for y in adj[x]:
                    affected.add((min(x,y), max(x,y)))
            
            for e in affected:
                eu, ev = e
                common = len(adj[eu] & adj[ev])
                books[e] = common
                books[(ev, eu)] = common
            
            new_global_max = get_global_max()
            
            accept = False
            if new_global_max < max_book:
                accept = True
            elif new_global_max == max_book:
                if random.random() < T:
                    accept = True
            
            if accept:
                max_book = new_global_max
                if max_book < best_max:
                    best_max = max_book
                    best_edge_set = set(edge_set)
                added = True
                T *= cooling
            else:
                # Revert
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                
                adj[a].discard(b)
                adj[b].discard(a)
                edge_set.discard((ka, kb))
                
                # Revert books for affected
                for e in affected:
                    eu, ev = e
                    common = len(adj[eu] & adj[ev])
                    books[e] = common
                    books[(ev, eu)] = common
                T *= cooling
            if added:
                break
                
        if time.time() - start_time > DEADLINE:
            break

    return [[u, v] for u, v in best_edge_set]
