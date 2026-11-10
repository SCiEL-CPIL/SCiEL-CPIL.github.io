import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(all_edges) # 120
    
    def creates_q3(u, v, adj):
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        mask_a = mask_u
        while mask_a:
            lsb_a = mask_a & -mask_a
            a = lsb_a.bit_length() - 1
            mask_a ^= lsb_a
            
            mask_b = mask_u & ~lsb_a
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b ^= lsb_b
                
                mask_c = mask_v & adj[a]
                mask_d = mask_v & adj[b]
                
                if not mask_c or not mask_d:
                    continue
                
                mask_c_iter = mask_c
                while mask_c_iter:
                    lsb_c = mask_c_iter & -mask_c_iter
                    c = lsb_c.bit_length() - 1
                    mask_c_iter ^= lsb_c
                    
                    if c == a or c == b:
                        continue
                        
                    mask_d_iter = mask_d & ~lsb_c
                    while mask_d_iter:
                        lsb_d = mask_d_iter & -mask_d_iter
                        d = lsb_d.bit_length() - 1
                        mask_d_iter ^= lsb_d
                        
                        if d == a or d == b or d == c:
                            continue
                        
                        mask_e = (adj[a] & adj[b]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if mask_e == 0:
                            continue
                            
                        mask_f = (adj[c] & adj[d]) & ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d))
                        if mask_f == 0:
                            continue
                        
                        if mask_e.bit_count() < mask_f.bit_count():
                            mask_e_iter = mask_e
                            while mask_e_iter:
                                lsb_e = mask_e_iter & -mask_e_iter
                                e = lsb_e.bit_length() - 1
                                mask_e_iter ^= lsb_e
                                if mask_f & adj[e] & ~lsb_e:
                                    return True
                        else:
                            mask_f_iter = mask_f
                            while mask_f_iter:
                                lsb_f = mask_f_iter & -mask_f_iter
                                f = lsb_f.bit_length() - 1
                                mask_f_iter ^= lsb_f
                                if mask_e & adj[f] & ~lsb_f:
                                    return True
        return False

    def greedy_construct():
        adj = [0] * n
        edges = []
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
        return adj, edges

    def local_search_add(adj, edges):
        improved = True
        while improved:
            improved = False
            missing_edges = [e for e in all_edges if not ((adj[e[0]] >> e[1]) & 1)]
            random.shuffle(missing_edges)
            
            for u, v in missing_edges:
                if time.time() > deadline - 5:
                    return adj, edges
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
                    improved = True
        return adj, edges

    def kick_and_fill(adj, edges, kick_size):
        current_adj = adj[:]
        current_edges = edges[:]
        
        for _ in range(kick_size):
            if not current_edges:
                break
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_edges.pop(idx)
            
        current_adj, current_edges = local_search_add(current_adj, current_edges)
        return current_adj, current_edges

    def polish_swaps(adj, edges):
        # Try single edge swaps: remove e1, add e2
        # If valid, keep it (size same), but might allow further additions?
        # This function just tries to find a swap that leads to a state where we can add more?
        # Or just find a valid swap to diversify?
        # For now, let's just do random kicks. Swaps are complex to evaluate for "potential".
        pass

    best_adj = [0] * n
    best_edges = []
    best_count = 0
    
    # Initial solution
    adj, edges = greedy_construct()
    adj, edges = local_search_add(adj, edges)
    best_adj = adj[:]
    best_edges = edges[:]
    best_count = len(edges)
    
    # Main loop
    while time.time() < deadline - 10:
        # Decide kick size
        # Larger kicks escape deeper local optima
        kick_size = random.randint(3, 15)
        
        # Occasionally restart from scratch
        if random.random() < 0.05:
            adj, edges = greedy_construct()
        else:
            adj, edges = kick_and_fill(best_adj, best_edges, kick_size)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_adj = adj[:]
            best_edges = edges[:]
            
    # Final polish: try many small kicks on the best solution
    while time.time() < deadline - 1:
        adj, edges = kick_and_fill(best_adj, best_edges, random.randint(1, 5))
        if len(edges) > best_count:
            best_count = len(edges)
            best_adj = adj[:]
            best_edges = edges[:]

    return [[u, v] for u, v in best_edges]
