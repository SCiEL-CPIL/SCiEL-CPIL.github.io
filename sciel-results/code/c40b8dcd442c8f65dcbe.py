import time
import random

def _is_bipartite(adj_mask):
    n = len(adj_mask)
    color = [-1] * n
    for i in range(n):
        if color[i] == -1:
            color[i] = 0
            stack = [i]
            while stack:
                u = stack.pop()
                m = adj_mask[u]
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        stack.append(v)
                    elif color[v] == color[u]:
                        return False
                    m ^= lsb
    return True

def _is_3_colorable(adj_mask):
    n = len(adj_mask)
    if _is_bipartite(adj_mask):
        return True
        
    colors = [-1] * n
    order = sorted(range(n), key=lambda i: bin(adj_mask[i]).count('1'), reverse=True)
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        used = 0
        m = adj_mask[v]
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            c = colors[u]
            if c != -1:
                used |= (1 << c)
            m ^= lsb
        for c in range(3):
            if not (used & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    adj = [0] * n
    
    # Clebsch
    for u in range(n):
        for v in range(u + 1, n):
            if bin(u ^ v).count('1') in (1, 4):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    current_edges = sum(bin(m).count('1') for m in adj) // 2
    best_edges = current_edges
    best_adj = adj[:]
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_tf_add(u, v):
        return (adj[u] & adj[v]) == 0
        
    def is_not_3c_add(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        res = not _is_3_colorable(adj)
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        return res
        
    def is_valid_swap(u1, v1, u2, v2):
        adj[u1] ^= (1 << v1)
        adj[v1] ^= (1 << u1)
        
        if adj[u2] & adj[v2]:
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            return False
            
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        
        res = not _is_3_colorable(adj)
        
        adj[u2] ^= (1 << v2)
        adj[v2] ^= (1 << u2)
        adj[u1] |= (1 << v1)
        adj[v1] |= (1 << u1)
        
        return res

    while time.time() < deadline:
        improved = False
        for u, v in all_edges:
            if time.time() > deadline: break
            if adj[u] & (1 << v): continue
            if not is_tf_add(u, v): continue
            if is_not_3c_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                improved = True
        
        if improved: continue
        
        # Swaps
        existing = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u] & (1 << v)]
        non_exist = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
        
        random.shuffle(existing)
        random.shuffle(non_exist)
        
        swap_found = False
        for e_out in existing:
            if time.time() > deadline: break
            if swap_found: break
            u1, v1 = e_out
            for e_in in non_exist:
                if time.time() > deadline: break
                u2, v2 = e_in
                if is_valid_swap(u1, v1, u2, v2):
                    adj[u1] ^= (1 << v1)
                    adj[v1] ^= (1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    swap_found = True
                    break
                    
        if swap_found: continue
        
        # Perturb
        if existing:
            u1, v1 = random.choice(existing)
            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            current_edges -= 1
        else:
            break

    edges = []
    for u in range(n):
        m = best_adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m ^= lsb
    return edges
