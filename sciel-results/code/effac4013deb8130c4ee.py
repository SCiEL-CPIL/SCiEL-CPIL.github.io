import time
import random

def construct():
    n = 40
    target_edges = 401
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Pre-compute all possible edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Initialize with random graph
    random.shuffle(all_pairs)
    initial_edges = all_pairs[:target_edges]
    
    # Adjacency bitsets
    adj = [0] * n
    for u, v in initial_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    current_edges = set(initial_edges)
    
    def get_max_book(adj, edges):
        max_b = 0
        for u, v in edges:
            b = (adj[u] & adj[v]).bit_count()
            if b > max_b:
                max_b = b
        return max_b

    current_max = get_max_book(adj, current_edges)
    best_max = current_max
    best_edges = set(current_edges)
    best_adj = adj[:]
    
    # Main optimization loop
    while time.time() < deadline:
        # Find the worst edge
        worst_edge = None
        worst_book = -1
        for u, v in current_edges:
            b = (adj[u] & adj[v]).bit_count()
            if b > worst_book:
                worst_book = b
                worst_edge = (u, v)
                
        if worst_edge is None:
            break
            
        u, v = worst_edge
        improved = False
        
        # Try to replace the worst edge with a random non-edge
        for _ in range(20): # Reduced attempts per worst edge to allow more iterations
            x, y = random.choice(all_pairs)
            if (x, y) in current_edges:
                continue
                
            # Tentatively swap
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges.remove((u,v))
            
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            current_edges.add((x,y))
            
            new_max = get_max_book(adj, current_edges)
            
            if new_max < current_max:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_edges = set(current_edges)
                    best_adj = adj[:]
                improved = True
                break
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u,v))
                
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
                current_edges.remove((x,y))
                
        if not improved:
            # Perform a random swap to explore
            # Pick a random edge to remove
            rem_u, rem_v = random.choice(list(current_edges))
            
            # Pick a random non-edge to add
            add_u = add_v = -1
            for _ in range(10):
                cand_u, cand_v = random.choice(all_pairs)
                if (cand_u, cand_v) not in current_edges:
                    add_u, add_v = cand_u, cand_v
                    break
            
            if add_u != -1:
                # Perform swap
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                current_edges.remove((rem_u, rem_v))
                
                adj[add_u] |= (1 << add_v)
                adj[add_v] |= (1 << add_u)
                current_edges.add((add_u, add_v))
                
                new_max = get_max_book(adj, current_edges)
                
                # Accept if not worse (plateau exploration)
                if new_max <= current_max:
                    current_max = new_max
                    if new_max < best_max:
                        best_max = new_max
                        best_edges = set(current_edges)
                        best_adj = adj[:]
                else:
                    # Revert
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
                    current_edges.add((rem_u, rem_v))
                    
                    adj[add_u] &= ~(1 << add_v)
                    adj[add_v] &= ~(1 << add_u)
                    current_edges.remove((add_u, add_v))

    return [list(e) for e in best_edges]
