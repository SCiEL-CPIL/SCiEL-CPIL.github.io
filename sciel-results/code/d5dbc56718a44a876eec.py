import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 16
    target_edges = 65
    
    # Helper to calculate score efficiently using bitmasks
    def calc_score(adj, degs):
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_score if max_sum < max_score else max_sum

    # Corrected calc_score (the above had a bug, let's use a clean version)
    def calc_score_clean(adj, degs):
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    def generate_seed_graph():
        """K_{8,8} plus one edge (0,1)"""
        adj = [0] * n
        degs = [0] * n
        edges = []
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
                edges.append([u, v])
        # Add edge (0,1)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        degs[0] += 1
        degs[1] += 1
        edges.append([0, 1])
        return adj, degs, edges

    def generate_random_graph():
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        edges = all_edges[:target_edges]
        
        adj = [0] * n
        degs = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1
        return adj, degs, edges

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    best_score = float('inf')
    best_adj = None
    best_degs = None
    best_edges = None

    # Start with the known good seed
    adj, degs, edges = generate_seed_graph()
    current_score = calc_score_clean(adj, degs)
    
    if current_score < best_score:
        best_score = current_score
        best_adj = adj[:]
        best_degs = degs[:]
        best_edges = get_edges_from_adj(adj)

    # Simulated Annealing / Local Search
    T = 5.0
    cooling = 0.9995
    no_improve_count = 0
    max_no_improve = 1000

    # Precompute list of all possible edges and non-edges for faster random selection?
    # Or just random sampling. With n=16, 120 pairs total.
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart or break
            break
            
        # Pick a random edge to remove
        u1, v1 = -1, -1
        while u1 == -1:
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 < v1 and (adj[u1] & (1 << v1)):
                break
            else:
                u1, v1 = -1, -1
                
        # Pick a random non-edge to add
        u2, v2 = -1, -1
        while u2 == -1:
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 < v2 and not (adj[u2] & (1 << v2)):
                break
            else:
                u2, v2 = -1, -1

        # Apply swap
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        degs[u1] -= 1
        degs[v1] -= 1
        
        adj[u2] |= (1 << v2)
        adj[v2] |= (1 << u2)
        degs[u2] += 1
        degs[v2] += 1
        
        new_score = calc_score_clean(adj, degs)
        
        delta = new_score - current_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_score = new_score
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
                best_degs = degs[:]
                best_edges = get_edges_from_adj(adj)
        else:
            # Revert
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            degs[u1] += 1
            degs[v1] += 1
            
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)
            degs[u2] -= 1
            degs[v2] -= 1
            
            no_improve_count += 1

        T *= cooling
        if T < 0.01:
            T = 0.01

    # If we exited due to no_improve, try a few random restarts if time remains
    while time.time() < deadline:
        adj, degs, _ = generate_random_graph()
        current_score = calc_score_clean(adj, degs)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
            best_degs = degs[:]
            best_edges = get_edges_from_adj(adj)
            
        # Brief local search from this random start
        T = 2.0
        for _ in range(500):
            if time.time() > deadline:
                break
            u1, v1 = -1, -1
            while u1 == -1:
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                if u1 < v1 and (adj[u1] & (1 << v1)):
                    break
                else:
                    u1, v1 = -1, -1
                    
            u2, v2 = -1, -1
            while u2 == -1:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 < v2 and not (adj[u2] & (1 << v2)):
                    break
                else:
                    u2, v2 = -1, -1

            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            degs[u1] -= 1
            degs[v1] -= 1
            
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            degs[u2] += 1
            degs[v2] += 1
            
            new_score = calc_score_clean(adj, degs)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj[:]
                    best_degs = degs[:]
                    best_edges = get_edges_from_adj(adj)
            else:
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                degs[u1] += 1
                degs[v1] += 1
                
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                degs[u2] -= 1
                degs[v2] -= 1
            
            T *= 0.999
            if T < 0.01:
                T = 0.01

    if best_edges is None:
        # Fallback
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return {'n': 16, 'edges': edges}
        
    return {'n': 16, 'edges': best_edges}
