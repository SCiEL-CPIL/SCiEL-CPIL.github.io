import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin
    
    n = 16
    m = 65
    
    # Bitmask representation
    # adj[i] is an integer where bit j is 1 if (i, j) is an edge
    adj = [0] * n
    deg = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1
        
    def has_edge(u, v):
        return (adj[u] >> v) & 1
        
    def _score():
        best = 0
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    # common neighbors of u and v
                    common = au & adj[v]
                    # iterate through set bits in common
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > best:
                            best = s
                        c ^= lsb
        return best

    # Initialize: K_{8,8} + one edge in A
    # A = 0..7, B = 8..15
    for u in range(8):
        for v in range(8, 16):
            add_edge(u, v)
    add_edge(0, 1)
    
    cur_score = _score()
    best_score = cur_score
    best_adj = adj[:]
    best_deg = deg[:]
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # State for local search
    # We track the set of edges to easily pick random edges/non-edges
    # But with bitmasks, we can just pick random u, v and check.
    # To pick a random edge:
    #   Pick random u, v. If edge, use it. Else retry.
    #   Prob of edge is 65/120 ~ 0.54. So ~2 attempts on average.
    
    def pick_random_edge():
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and has_edge(u, v):
                return u, v
                
    def pick_random_non_edge():
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not has_edge(u, v):
                return u, v

    no_improve = 0
    max_no_improve = 2000
    T = 5.0
    cool_rate = 0.999
    
    while time.time() - t0 < DEADLINE:
        # Pick swap
        u1, v1 = pick_random_edge()
        u2, v2 = pick_random_non_edge()
        
        # Perform swap
        remove_edge(u1, v1)
        add_edge(u2, v2)
        
        new_score = _score()
        
        if new_score < cur_score:
            cur_score = new_score
            no_improve = 0
            if cur_score < best_score:
                best_score = cur_score
                best_adj = adj[:]
                best_deg = deg[:]
        else:
            # Simulated Annealing acceptance
            delta = new_score - cur_score
            # Accept if delta is small and random
            if delta <= 1 and random.random() < 0.1:
                cur_score = new_score
                no_improve += 1
            else:
                # Revert
                remove_edge(u2, v2)
                add_edge(u1, v1)
                no_improve += 1
        
        T *= cool_rate
        if T < 0.5:
            T = 0.5
            
        if no_improve > max_no_improve:
            # Restart with perturbation
            # Do 5 random swaps
            for _ in range(5):
                eu1, ev1 = pick_random_edge()
                eu2, ev2 = pick_random_non_edge()
                remove_edge(eu1, ev1)
                add_edge(eu2, ev2)
            cur_score = _score()
            no_improve = 0
            T = 5.0
            max_no_improve = 2000
            
    # Reconstruct edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
