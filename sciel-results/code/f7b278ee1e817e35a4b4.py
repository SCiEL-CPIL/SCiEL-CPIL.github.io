import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def try_greedy(order_seed):
        random.seed(order_seed)
        masks = [0] * n
        edges_list = []
        
        # Shuffle edges
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            # Check if (u, v) can be added
            common_uv = (masks[u] & masks[v]).bit_count()
            if common_uv >= 3:
                continue
                
            valid = True
            
            # Check neighbors of v
            mv = masks[v]
            while mv:
                lsb = mv & -mv
                w = lsb.bit_length() - 1
                mv ^= lsb
                if (masks[u] & masks[w]).bit_count() >= 2:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check neighbors of u
            mu = masks[u]
            while mu:
                lsb = mu & -mu
                w = lsb.bit_length() - 1
                mu ^= lsb
                if (masks[v] & masks[w]).bit_count() >= 2:
                    valid = False
                    break
            if not valid:
                continue
            
            # Add edge
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            edges_list.append([u, v])
            
        return edges_list, len(edges_list)

    seed = 0
    while time.time() < deadline:
        edges, count = try_greedy(seed)
        if count > best_count:
            best_count = count
            best_edges = edges
        seed += 1
        
    return best_edges
