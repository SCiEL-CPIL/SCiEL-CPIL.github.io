import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 15
    
    def can_add(u, v, adj, deg):
        du_new = deg[u] + 1
        dv_new = deg[v] + 1
        
        # Case 1: s-u-v-t
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        s = mask_u
        while s:
            lsb_s = s & -s
            i_s = lsb_s.bit_length() - 1
            s -= lsb_s
            ds = deg[i_s]
            
            t = mask_v & ~(1 << i_s)
            while t:
                lsb_t = t & -t
                i_t = lsb_t.bit_length() - 1
                t -= lsb_t
                if ds == deg[i_t]:
                    return False
        
        # Case 2: u-v-x-y
        x = mask_v
        while x:
            lsb_x = x & -x
            i_x = lsb_x.bit_length() - 1
            x -= lsb_x
            mask_y = adj[i_x] & ~(1 << u) & ~(1 << v)
            y = mask_y
            while y:
                lsb_y = y & -y
                i_y = lsb_y.bit_length() - 1
                y -= lsb_y
                if du_new == deg[i_y]:
                    return False
        
        # Case 3: x-y-v-u
        x = mask_u
        while x:
            lsb_x = x & -x
            i_x = lsb_x.bit_length() - 1
            x -= lsb_x
            dx = deg[i_x]
            mask_y = adj[i_x] & ~(1 << u) & ~(1 << v)
            y = mask_y
            while y:
                lsb_y = y & -y
                i_y = lsb_y.bit_length() - 1
                y -= lsb_y
                if dx == du_new:
                    return False
        
        return True

    def add_edge(u, v, adj, deg):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1

    def remove_edge(u, v, adj, deg):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1

    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
            
    best_edges = []
    best_count = 0
    
    num_restarts = 1000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        adj = [0] * n
        deg = [0] * n
        
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
            if can_add(u, v, adj, deg):
                add_edge(u, v, adj, deg)
        
        current_count = sum(deg) // 2
        
        improved = True
        max_swap_iters = 10
        for _ in range(max_swap_iters):
            if time.time() > deadline:
                break
            improved = False
            
            non_edges = []
            for u, v in all_edges:
                if not (adj[u] & (1 << v)):
                    non_edges.append((u, v))
            
            if not non_edges:
                break
                
            random.shuffle(non_edges)
            
            current_edges_list = []
            for u in range(n):
                v_mask = adj[u]
                while v_mask:
                    lsb = v_mask & -v_mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        current_edges_list.append((u, v))
                    v_mask -= lsb
            
            if not current_edges_list:
                break
                
            random.shuffle(current_edges_list)
            
            for rem_u, rem_v in current_edges_list:
                if time.time() > deadline:
                    break
                
                remove_edge(rem_u, rem_v, adj, deg)
                
                added = False
                for add_u, add_v in non_edges:
                    if can_add(add_u, add_v, adj, deg):
                        add_edge(add_u, add_v, adj, deg)
                        improved = True
                        added = True
                        break
                    
                if not added:
                    add_edge(rem_u, rem_v, adj, deg)
                
                if improved:
                    break
                    
            if not improved:
                break
        
        if sum(deg) // 2 > best_count:
            best_count = sum(deg) // 2
            best_edges = []
            for u in range(n):
                v_mask = adj[u]
                while v_mask:
                    lsb = v_mask & -v_mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        best_edges.append([u, v])
                    v_mask -= lsb

    return best_edges
