import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    random.seed(42)
    
    n = 18
    full_mask = (1 << n) - 1
    
    # Precompute all possible triangles
    all_tris = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                all_tris.append((1 << u) | (1 << v) | (1 << w))
    
    # Precompute supersets for each triangle
    supersets = {}
    for t in all_tris:
        rest = full_mask ^ t
        subs = []
        sub = rest
        while True:
            subs.append(t | sub)
            if sub == 0:
                break
            sub = (sub - 1) & rest
        supersets[t] = np.array(subs, dtype=np.int32)
        
    # Precompute popcounts
    popcounts = np.array([s.bit_count() for s in range(1 << n)], dtype=np.int32)
    
    # Initialize graph as K_{3,3,3,3,3,3}
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if i // 3 != j // 3:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def get_present_tris():
        tris = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    w_mask = common
                    while w_mask:
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        if u < v < w:
                            tris.append((1 << u) | (1 << v) | (1 << w))
                        w_mask ^= lsb
        return tris

    def score():
        present_tris = get_present_tris()
        bad_np = np.zeros(1 << n, dtype=bool)
        for t in present_tris:
            bad_np[supersets[t]] = True
        valid = ~bad_np
        if not np.any(valid):
            return 0
        return popcounts[valid].max()

    def can_add_edge(u, v):
        if adj[u] & (1 << v):
            return False
        common = adj[u] & adj[v]
        c_mask = common
        while c_mask:
            lsb = c_mask & -c_mask
            w = lsb.bit_length() - 1
            if adj[w] & common:
                return False
            c_mask ^= lsb
        return True

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    current_score = score()
    best_score = current_score
    best_adj = adj[:]
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for restart in range(20):
        if time.time() > deadline - 20:
            break
            
        if restart > 0:
            adj = best_adj[:]
            for _ in range(30):
                u, v = random.choice(all_edges)
                if adj[u] & (1 << v):
                    remove_edge(u, v)
                else:
                    if can_add_edge(u, v):
                        add_edge(u, v)
            current_score = score()
        
        for step in range(2000):
            if time.time() > deadline - 5:
                break
            
            u, v = random.choice(all_edges)
            old_score = current_score
            
            if adj[u] & (1 << v):
                remove_edge(u, v)
                new_score = score()
                if new_score <= old_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj[:]
                else:
                    add_edge(u, v)
            else:
                if can_add_edge(u, v):
                    add_edge(u, v)
                    new_score = score()
                    if new_score <= old_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_adj = adj[:]
                    else:
                        remove_edge(u, v)
        
    # Verify best_adj is K4-free
    # If not, use initial
    final_adj = best_adj
    # Quick K4 check
    k4_free = True
    for u in range(n):
        for v in range(u + 1, n):
            if final_adj[u] & (1 << v):
                common = final_adj[u] & final_adj[v]
                c_mask = common
                while c_mask:
                    lsb = c_mask & -c_mask
                    w = lsb.bit_length() - 1
                    if final_adj[w] & common:
                        k4_free = False
                        break
                    c_mask ^= lsb
                if not k4_free:
                    break
        if not k4_free:
            break
    
    if not k4_free:
        final_adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if i // 3 != j // 3:
                    final_adj[i] |= (1 << j)
                    final_adj[j] |= (1 << i)

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if final_adj[u] & (1 << v):
                edges.append([u, v])
    return edges
