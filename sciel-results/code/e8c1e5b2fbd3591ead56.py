import random
import time
import math

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Helper to calculate score from bitmask adjacency
    def calc_score(adj):
        # adj is a list of integers (bitmasks)
        degs = [bin(a).count('1') for a in adj]
        max_sum = 0
        # Iterate over all pairs u < v
        for u in range(n):
            if not adj[u]: continue
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # u and v are connected
                    common = adj[u] & adj[v]
                    # Iterate over common neighbors w
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    def generate_baseline():
        # K_{8,8} plus edge (0,1)
        adj = [0] * n
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        return adj

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    best_adj = generate_baseline()
    best_score = calc_score(best_adj)
    
    # Precompute list of all possible edges and non-edges for faster sampling?
    # Or just random sampling.
    
    def random_swap(adj):
        # Pick random edge to remove
        while True:
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 < v1 and (adj[u1] & (1 << v1)):
                break
        
        # Pick random non-edge to add
        while True:
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 < v2 and not (adj[u2] & (1 << v2)):
                break
        
        return (u1, v1), (u2, v2)

    # Local Search
    current_adj = best_adj[:]
    current_score = best_score
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart from best or random?
            # Restart from best is safer.
            current_adj = best_adj[:]
            current_score = best_score
            no_improve = 0
            # Optional: kick current_adj slightly
            for _ in range(10):
                if time.time() > deadline: break
                e_rm, e_add = random_swap(current_adj)
                u1, v1 = e_rm
                u2, v2 = e_add
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
            current_score = calc_score(current_adj)
            continue

        e_rm, e_add = random_swap(current_adj)
        u1, v1 = e_rm
        u2, v2 = e_add
        
        # Apply swap
        current_adj[u1] &= ~(1 << v1)
        current_adj[v1] &= ~(1 << u1)
        current_adj[u2] |= (1 << v2)
        current_adj[v2] |= (1 << u2)
        
        new_score = calc_score(current_adj)
        
        if new_score <= current_score:
            current_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
        else:
            # Revert
            current_adj[u1] |= (1 << v1)
            current_adj[v1] |= (1 << u1)
            current_adj[u2] &= ~(1 << v2)
            current_adj[v2] &= ~(1 << u2)
            no_improve += 1
            
            # Small chance to accept worse move to escape local minima
            if random.random() < 0.05:
                # Re-apply swap
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                current_score = new_score
                no_improve = 0
                # Note: we do NOT update best here unless it's better, but we keep this state

    return {'n': n, 'edges': get_edges(best_adj)}
