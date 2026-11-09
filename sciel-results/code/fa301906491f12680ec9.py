import numpy as np
import random
import time

def construct():
    # Try to find a valid graph using Local Search (Greedy Repair)
    # This approach explores the general graph space, unlike the Circulant approach.
    
    # We aim for the known record of 35. If we fail, we drop down.
    for n in range(35, 0, -1):
        # Try multiple times for each n to handle stochasticity
        for attempt in range(20):
            result = _local_search(n)
            if result is not None:
                return result
            
    # Fallback: Paley graph of order 17
    # This is a known K4-free graph with alpha=5
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def _local_search(n):
    # Initialize a random graph
    p = 0.5
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    max_iters = 5000
    start_time = time.time()
    timeout = 5.0 # 5 seconds per attempt
    
    for _ in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        # Check for K4
        k4_found = _find_one_k4(adj, n)
        if k4_found:
            u, v, w, x = k4_found
            edges_in_k4 = [(u, v), (u, w), (u, x), (v, w), (v, x), (w, x)]
            ru, rv = random.choice(edges_in_k4)
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
            continue
            
        # Check for Independent 6-set
        indep6_found = _find_one_indep6(adj, n)
        if indep6_found:
            u, v = random.sample(indep6_found, 2)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
            
        # No K4 and no Indep6 found
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    return None

def _find_one_k4(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                continue
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            c = common
            while c:
                lsb_w = c & -c
                w = lsb_w.bit_length() - 1
                c ^= lsb_w
                
                neighbors_in_common = adj[w] & common
                neighbors_in_common &= ~(1 << w)
                
                if neighbors_in_common:
                    lsb_x = neighbors_in_common & -neighbors_in_common
                    x = lsb_x.bit_length() - 1
                    return (u, v, w, x)
                    
    return None

def _find_one_indep6(adj, n):
    comp_adj = [0] * n
    mask_all = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = mask_all ^ (1 << i) ^ adj[i]
        
    def find_clique(mask, k, current_clique):
        if k == 0:
            return current_clique
        if k == 1:
            if mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                return current_clique + [v]
            return None
            
        # Get list of vertices in mask
        # Order by degree in mask (descending) to find clique faster
        vertices = []
        c = mask
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            # Degree of v in the induced subgraph on mask
            deg = bin(mask & comp_adj[v]).count('1')
            vertices.append((deg, v))
            
        # Sort by degree descending
        vertices.sort(reverse=True)
        
        for deg, v in vertices:
            # If degree is less than k-1, we can't form a clique of size k including v
            if deg < k - 1:
                continue
                
            next_mask = mask & comp_adj[v]
            next_mask &= ~(1 << v)
            
            if next_mask:
                res = find_clique(next_mask, k - 1, current_clique + [v])
                if res is not None:
                    return res
                    
        return None

    return find_clique(mask_all, 6, [])

