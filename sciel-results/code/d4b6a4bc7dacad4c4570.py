import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin
    
    n = 16
    m = 65
    
    # Bitmask representation
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
        """Compute max triangle degree sum."""
        best = 0
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    common = au & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > best:
                            best = s
                        c ^= lsb
        return best
    
    def _triangles_list():
        """Return list of all triangles as sorted tuples (u,v,w) with u<v<w."""
        tris = []
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    common = au & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        tris.append((u, v, w))
                        c ^= lsb
        return tris
    
    def _score_from_triangles(tris):
        """Compute max degree sum over given triangle list."""
        best = 0
        for (u, v, w) in tris:
            s = deg[u] + deg[v] + deg[w]
            if s > best:
                best = s
        return best
    
    def _delta_score_swap(u1, v1, u2, v2):
        """
        Compute exact new score after removing edge (u1,v1) and adding edge (u2,v2).
        Uses incremental update: only triangles containing the affected edges change.
        Returns the new score (or -1 if the swap is invalid, i.e., u2,v2 already an edge).
        """
        if has_edge(u2, v2):
            return -1
        
        # Triangles that will be destroyed (contain edge u1-v1)
        destroyed = []
        common_uv1 = adj[u1] & adj[v1]
        c = common_uv1
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            destroyed.append((u1, v1, w))
            c ^= lsb
        
        # Triangles that will be created (contain edge u2-v2, i.e., common neighbors of u2 and v2)
        created = []
        common_uv2 = adj[u2] & adj[v3] if False else adj[u2] & adj[v2]
        # But we need to be careful: some of these might also contain u1-v1
        c = common_uv2
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            created.append((u2, v2, w))
            c ^= lsb
        
        # All other triangles are unaffected. We need the max over:
        # 1. All triangles NOT in destroyed (and not in created yet)
        # 2. All triangles in created
        # The tricky part: triangles in both destroyed and created need careful handling.
        
        # Simpler approach: compute the score of all triangles except destroyed,
        # then add created triangles.
        # But we need the full set. Let's just recompute from scratch but faster.
        # Actually, let's just compute the score properly.
        
        # Perform the swap temporarily
        remove_edge(u1, v1)
        add_edge(u2, v2)
        
        new_s = _score()
        
        # Revert
        remove_edge(u2, v2)
        add_edge(u1, v1)
        
        return new_s
    
    def _penalty_score():
        """Smooth surrogate: sum of squares of triangle degree sums above threshold."""
        total = 0.0
        threshold = 20
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    common = au & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > threshold:
                            total += (s - threshold) ** 2
                        c ^= lsb
        return total
    
    def _init_bipartite():
        """Initialize with K_{8,8} plus one intra-partition edge."""
        for i in range(n):
            adj[i] = 0
            deg[i] = 0
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        add_edge(0, 1)
    
    def _init_random_balanced():
        """Initialize with a random graph trying to balance degrees."""
        for i in range(n):
            adj[i] = 0
            deg[i] = 0
        
        # Start with bipartite structure
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        
        # Now we have 64 edges, need 1 more
        # Add a random edge within one partition
        side = random.choice([0, 1])
        if side == 0:
            a, b = random.sample(range(8), 2)
        else:
            a, b = random.sample(range(8, 16), 2)
        add_edge(a, b)
    
    def _local_search_polish(adj_in, deg_in, time_limit):
        """
        Greedy local search: try all possible single edge swaps and take the best improvement.
        Repeat until no improvement or time limit.
        """
        # Work on a copy
        adj_ls = adj_in[:]
        deg_ls = deg_in[:]
        
        def ls_has_edge(u, v):
            return (adj_ls[u] >> v) & 1
        
        def ls_add(u, v):
            adj_ls[u] |= (1 << v)
            adj_ls[v] |= (1 << u)
            deg_ls[u] += 1
            deg_ls[v] += 1
        
        def ls_remove(u, v):
            adj_ls[u] &= ~(1 << v)
            adj_ls[v] &= ~(1 << u)
            deg_ls[u] -= 1
            deg_ls[v] -= 1
        
        def ls_score():
            best = 0
            for u in range(n):
                au = adj_ls[u]
                for v in range(u + 1, n):
                    if (au >> v) & 1:
                        common = au & adj_ls[v]
                        c = common
                        while c:
                            lsb = c & -c
                            w = lsb.bit_length() - 1
                            s = deg_ls[u] + deg_ls[v] + deg_ls[w]
                            if s > best:
                                best = s
                            c ^= lsb
            return best
        
        cur_s = ls_score()
        improved = True
        t_start = time.time()
        
        while improved and (time.time() - t_start) < time_limit:
            improved = False
            best_delta = 0
            best_swap = None
            
            # Try all possible swaps
            for u1 in range(n):
                for v1 in range(u1 + 1, n):
                    if not ls_has_edge(u1, v1):
                        continue
                    for u2 in range(n):
                        if u2 == u1 or u2 == v1:
                            continue
                        for v2 in range(u2 + 1, n):
                            if v2 == u1 or v2 == v1:
                                continue
                            if ls_has_edge(u2, v2):
                                continue
                            
                            # Try swap
                            ls_remove(u1, v1)
                            ls_add(u2, v2)
                            new_s = ls_score()
                            ls_remove(u2, v2)
                            ls_add(u1, v1)
                            
                            if new_s < cur_s:
                                best_delta = cur_s - new_s
                                best_swap = (u1, v1, u2, v2)
            
            if best_swap is not None:
                u1, v1, u2, v2 = best_swap
                ls_remove(u1, v1)
                ls_add(u2, v2)
                cur_s -= best_delta
                improved = True
        
        return adj_ls, deg_ls, cur_s
    
    # Initialize
    _init_bipartite()
    
    cur_score = _score()
    best_score = cur_score
    best_adj = adj[:]
    best_deg = deg[:]
    
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
    
    # Simulated annealing with exact score
    no_improve = 0
    T = 3.0
    cool_rate = 0.9998
    max_no_improve = 2000
    
    # Precompute edge list for faster random selection
    def get_edge_list():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if has_edge(u, v):
                    edges.append((u, v))
        return edges
    
    def get_non_edge_list():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not has_edge(u, v):
                    edges.append((u, v))
        return edges
    
    while time.time() - t0 < DEADLINE - 10.0:  # Leave 10s for polish
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
            delta = new_score - cur_score
            if delta <= 1 and random.random() < 0.1:
                cur_score = new_score
                no_improve += 1
            else:
                remove_edge(u2, v2)
                add_edge(u1, v1)
                no_improve += 1
        
        T *= cool_rate
        if T < 0.3:
            T = 0.3
            
        if no_improve > max_no_improve:
            # Restart with perturbation
            for _ in range(3):
                eu1, ev1 = pick_random_edge()
                eu2, ev2 = pick_random_non_edge()
                remove_edge(eu1, ev1)
                add_edge(eu2, ev2)
            cur_score = _score()
            no_improve = 0
            T = 3.0
            max_no_improve = 2000
    
    # Final local search polish on best solution
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 2.0:
        polish_time = min(remaining_time - 1.0, 8.0)
        best_adj, best_deg, polish_score = _local_search_polish(best_adj, best_deg, polish_time)
        if polish_score < best_score:
            best_score = polish_score
    
    # Reconstruct edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
