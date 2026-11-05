import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18

    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True

    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)

    def largest_tf(adj, time_limit=None):
        best = [0]
        full_mask = (1 << n) - 1
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        
        # Heuristic 1: descending degree
        for order_mode in ['desc', 'asc']:
            if order_mode == 'desc':
                order = sorted(range(n), key=lambda x: -degrees[x])
            else:
                order = sorted(range(n), key=lambda x: degrees[x])
            
            greedy_set = 0
            for v in order:
                neighbors_in_set = adj[v] & greedy_set
                if not neighbors_in_set:
                    greedy_set |= (1 << v)
                    continue
                ok = True
                g = neighbors_in_set
                while g:
                    lsb1 = g & -g
                    x = lsb1.bit_length() - 1
                    g ^= lsb1
                    if adj[x] & adj[v] & greedy_set & ~lsb1:
                        ok = False
                        break
                if ok:
                    greedy_set |= (1 << v)
            
            sz = bin(greedy_set).count('1')
            if sz > best[0]:
                best[0] = sz
        
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            c_size = bin(C_mask).count('1')
            if sz + c_size <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return

            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = bin(adj[vid] & C_mask).count('1')
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
                
            v = chosen_v
            v_bit = 1 << v
            
            bnb(S_mask, C_mask & ~v_bit)
            
            neighbors_in_s = adj[v] & S_mask
            can_add = True
            if neighbors_in_s:
                g = neighbors_in_s
                while g:
                    lsb1 = g & -g
                    x = lsb1.bit_length() - 1
                    g ^= lsb1
                    if adj[x] & adj[v] & S_mask & ~lsb1:
                        can_add = False
                        break
            if can_add:
                bnb(S_mask | v_bit, C_mask & ~v_bit)
                
        bnb(0, full_mask)
        return best[0]

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    random.seed(42)
    best_adj = None
    best_score = n
    
    # Start with complete 3-partite graph T(18,3)
    # Parts: 0-5, 6-11, 12-17
    # Edges between different parts
    initial_adj = [0] * n
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for p1 in range(3):
        for p2 in range(p1+1, 3):
            for u in parts[p1]:
                for v in parts[p2]:
                    initial_adj[u] |= (1 << v)
                    initial_adj[v] |= (1 << u)
    
    initial_score = largest_tf(initial_adj, time_limit=deadline - 10.0)
    if initial_score < best_score:
        best_score = initial_score
        best_adj = initial_adj[:]

    def generate_random_k4_free():
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                current_edges = sum(bin(adj[i]).count('1') for i in range(n)) // 2
                total_possible = n * (n - 1) // 2
                density = current_edges / total_possible
                prob = max(0.3, 1.0 - density)
                if random.random() < prob:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # Local Search
    while time.time() < deadline - 15.0:
        adj = best_adj[:] if best_adj else generate_random_k4_free()
        score = largest_tf(adj, time_limit=deadline - 10.0)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]

        # Update degrees for heuristic
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        
        improved = True
        max_local_iters = 20
        local_iter = 0
        while improved and time.time() < deadline - 5.0 and local_iter < max_local_iters:
            local_iter += 1
            improved = False
            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not ((adj[i] >> j) & 1)]
            non_edges.sort(key=lambda e: -(degrees[e[0]] + degrees[e[1]]))
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline - 5.0:
                    break
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                    new_score = largest_tf(adj, time_limit=deadline - 10.0)
                    
                    if new_score < score:
                        score = new_score
                        improved = True
                        if score < best_score:
                            best_score = score
                            best_adj = adj[:]
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    
                    if improved:
                        break

    if best_adj is None:
        best_adj = initial_adj

    edges = get_edge_list(best_adj)
    return edges
