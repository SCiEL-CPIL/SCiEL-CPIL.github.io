import time
import random

def construct():
    N = 40
    adj = [0] * N
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def has_edge(u, v):
        return (adj[u] >> v) & 1
        
    def count_edges():
        return sum(bin(a).count('1') for a in adj) // 2
        
    def creates_c6(u, v):
        if has_edge(u, v):
            return False
            
        mask_uv = (1 << u) | (1 << v)
        
        a_mask = adj[u]
        while a_mask:
            a = (a_mask & -a_mask).bit_length() - 1
            a_mask &= a_mask - 1
            
            d_mask = adj[v]
            while d_mask:
                d = (d_mask & -d_mask).bit_length() - 1
                d_mask &= d_mask - 1
                
                if d == a:
                    continue
                
                common = adj[a] & adj[d]
                common &= ~mask_uv
                common &= ~(1 << a)
                common &= ~(1 << d)
                
                if common:
                    return True
                    
        return False

    best_edges = 0
    best_adj = [0] * N
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Use a global counter for random seed variation
    restart_count = 0
    
    while time.time() < deadline:
        restart_count += 1
        adj = [0] * N
        # Vary the random seed for each restart to explore different orderings
        random.seed(restart_count * 12345 + 67890)
        
        all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not creates_c6(u, v):
                add_edge(u, v)
        
        current_edges = count_edges()
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            
        # Local Search: Try to improve
        # We allow local search to run for a short fixed time or a few iterations
        local_start = time.time()
        local_deadline = local_start + 5.0 # 5 seconds per restart for local search
        # If we are close to the global deadline, skip local search
        if deadline - time.time() < 10.0:
            continue
            
        no_improve = 0
        while time.time() < local_deadline and time.time() < deadline:
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u == v or has_edge(u, v):
                continue
                
            if not creates_c6(u, v):
                add_edge(u, v)
                current_edges += 1
                no_improve = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Try to resolve conflict
                mask_uv = (1 << u) | (1 << v)
                a_found = -1
                d_found = -1
                x_found = -1
                found = False
                
                a_mask = adj[u]
                while a_mask and not found:
                    a = (a_mask & -a_mask).bit_length() - 1
                    a_mask &= a_mask - 1
                    
                    d_mask = adj[v]
                    while d_mask and not found:
                        d = (d_mask & -d_mask).bit_length() - 1
                        d_mask &= d_mask - 1
                        
                        if d == a:
                            continue
                        
                        common = adj[a] & adj[d]
                        common &= ~mask_uv
                        common &= ~(1 << a)
                        common &= ~(1 << d)
                        
                        if common:
                            x = (common & -common).bit_length() - 1
                            a_found = a
                            d_found = d
                            x_found = x
                            found = True
                            
                if found:
                    candidates = [(u, a_found), (a_found, x_found), (x_found, d_found), (d_found, v)]
                    random.shuffle(candidates)
                    
                    improved = False
                    for ru, rv in candidates:
                        if not has_edge(ru, rv):
                            continue
                        
                        remove_edge(ru, rv)
                        current_edges -= 1
                        
                        if not creates_c6(u, v):
                            add_edge(u, v)
                            current_edges += 1
                            improved = True
                            no_improve = 0
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = adj[:]
                            break
                        else:
                            add_edge(ru, rv)
                            current_edges += 1
                            
                    if not improved:
                        no_improve += 1
                        if no_improve > 50:
                            break
                else:
                    no_improve += 1
                    if no_improve > 50:
                        break

    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
