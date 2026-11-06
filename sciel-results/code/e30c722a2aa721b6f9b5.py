import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    n = 40

    def would_create_c6(adj, u, v):
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)
        
        # Iterate over neighbors of u
        a_mask = Nu
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a
            
            # Iterate over neighbors of v
            d_mask = Nv
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d
                
                if a == d:
                    continue
                
                # We need a path a-b-c-d in G \ {u, v, a, d}
                # b in N(a), c in N(d), (b, c) in E
                # b != c, b not in {u, v, a, d}, c not in {u, v, a, d}
                
                mask_b = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                mask_c = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                
                # Check if there is an edge between mask_b and mask_c
                # Iterate over b in mask_b
                b_temp = mask_b
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & mask_c:
                        return True
                        
        return False

    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def run_greedy(start_edges, seed_val):
        random.seed(seed_val)
        adj = build_adj(start_edges)
        edges = set(start_edges)
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline:
                return edges
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return edges

    def local_search(edges_set, seed_val):
        random.seed(seed_val)
        adj = build_adj(edges_set)
        current_edges = set(edges_set)
        
        # Pass 1: Add any possible non-edge
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline - 2:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
        
        # Pass 2: Swaps
        removable = list(current_edges)
        random.shuffle(removable)
        
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        random.shuffle(non_edges)
        
        consecutive_failures = 0
        max_failures = 50
        
        while time.time() < deadline - 2:
            if consecutive_failures > max_failures:
                break
            if not removable:
                break
            
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            
            old_adj = adj[:]
            old_edges = set(current_edges)
            
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(e_rem)
            
            added = 0
            local_non_edges = non_edges[:]
            random.shuffle(local_non_edges)
            
            for u, v in local_non_edges:
                if added >= 2:
                    break
                if time.time() > deadline - 2:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            
            if added >= 2:
                consecutive_failures = 0
                removable = list(current_edges)
                random.shuffle(removable)
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] & (1 << j)):
                            non_edges.append((i, j))
                random.shuffle(non_edges)
            else:
                adj = old_adj
                current_edges = old_edges
                consecutive_failures += 1

        return current_edges

    def perturb_and_refine(edges_set, seed_val, num_remove):
        random.seed(seed_val)
        edges = set(edges_set)
        to_remove = random.sample(list(edges), min(num_remove, len(edges)))
        for e in to_remove:
            edges.remove(e)
        
        adj = build_adj(edges)
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline - 5:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        edges = local_search(edges, seed_val)
        return edges

    # Main search loop
    # Start with a random sparse graph or K5 seed
    best_edges = set()
    # Let's try a few different seeds
    for seed in [42, 123, 999]:
        if time.time() > deadline - 10:
            break
        # Random initial graph
        edges = set()
        random.seed(seed)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.1:
                    edges.add((i, j))
        
        best_candidate = run_greedy(edges, seed)
        best_candidate = local_search(best_candidate, seed)
        
        if len(best_candidate) > len(best_edges):
            best_edges = best_candidate
            
        # Perturb and refine
        for i in range(10):
            if time.time() > deadline - 10:
                break
            perturbed = perturb_and_refine(best_edges, seed + i * 100, random.randint(2, 5))
            if len(perturbed) > len(best_edges):
                best_edges = perturbed

    return [list(e) for e in best_edges]
