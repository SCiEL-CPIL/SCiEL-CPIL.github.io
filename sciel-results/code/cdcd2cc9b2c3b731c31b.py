import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    n = 20
    
    def is_valid_addition(u, v, adj):
        # Check if adding (u, v) creates a K3,3
        nv = adj[v] & ~(1 << u)
        nu = adj[u] & ~(1 << v)
        
        verts_nv = []
        temp = nv
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts_nv.append(idx)
            temp ^= lsb
            
        if len(verts_nv) < 2:
            return True
            
        for i in range(len(verts_nv)):
            a2 = verts_nv[i]
            for j in range(i+1, len(verts_nv)):
                a3 = verts_nv[j]
                common = adj[a2] & adj[a3]
                common_in_nu = common & nu
                common_in_nu &= ~(1 << a2)
                common_in_nu &= ~(1 << a3)
                if common_in_nu & (common_in_nu - 1):
                    return False
        return True

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            v_bit = adj[u]
            while v_bit:
                lsb = v_bit & -v_bit
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                v_bit ^= lsb
        return edges

    best_edges = 0
    best_adj = [0] * n

    while time.time() < deadline:
        adj = [0] * n
        edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if is_valid_addition(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Local Search
        while time.time() < deadline:
            all_edges = []
            for u in range(n):
                v_bit = adj[u]
                while v_bit:
                    lsb = v_bit & -v_bit
                    v = lsb.bit_length() - 1
                    if u < v:
                        all_edges.append((u, v))
                    v_bit ^= lsb
            
            if not all_edges:
                break
                
            u_rem, v_rem = random.choice(all_edges)
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            
            non_edges = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
            
            if not non_edges:
                break
            
            found_add = False
            for _ in range(5):
                if time.time() > deadline:
                    break
                u_add, v_add = random.choice(non_edges)
                if is_valid_addition(u_add, v_add, adj):
                    adj[u_add] |= (1 << v_add)
                    adj[v_add] |= (1 << u_add)
                    found_add = True
                    break
            
            if not found_add:
                adj[u_rem] |= (1 << v_rem)
                adj[v_rem] |= (1 << u_rem)

        current_edges = count_edges(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    return edges_from_adj(best_adj)
