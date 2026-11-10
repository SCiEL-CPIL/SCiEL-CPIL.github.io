import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute conflict masks
    # conflict_mask[u] has bit v set if (u-v)%m is in forbidden (and u!=v)
    conflict_mask = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u == v:
                continue
            if (u - v) % m in forbidden:
                mask |= (1 << v)
        conflict_mask[u] = mask
        
    # Precompute degrees (number of conflicts) for greedy ordering
    degrees = [bin(mask).count('1') for mask in conflict_mask]
    
    start_time = time.time()
    timeout = 590.0 # Safety margin
    
    best_mask = 0
    best_size = 0
    
    # Helper: Greedy build given an order
    def greedy_build(order):
        current_mask = 0
        for v in order:
            if current_mask & conflict_mask[v]:
                continue
            current_mask |= (1 << v)
        return current_mask
    
    # Helper: Hill Climb (Add only)
    def hill_climb(mask):
        improved = True
        while improved:
            improved = False
            # Find any vertex that can be added
            # To make it stochastic and escape symmetries, we can shuffle candidates
            # But simple iteration is fast. Let's try random selection among valid adds if we want diversity, 
            # but for speed, just finding one is enough.
            # However, which one we pick matters for future adds.
            # Let's pick a random valid add to explore different paths.
            valid_adds = []
            for v in range(m):
                if (mask >> v) & 1:
                    continue
                if mask & conflict_mask[v]:
                    continue
                valid_adds.append(v)
            
            if valid_adds:
                v = random.choice(valid_adds)
                mask |= (1 << v)
                improved = True
                
        return mask
    
    # Initial best solution
    # Use a few different greedy strategies
    for _ in range(20):
        order = list(range(m))
        # Sort by degree with some randomness
        # Add noise to degree to vary the order
        noisy_degrees = [d + random.uniform(0, 10) for d in degrees]
        order.sort(key=lambda x: noisy_degrees[x])
        
        curr_mask = greedy_build(order)
        curr_mask = hill_climb(curr_mask)
        curr_size = bin(curr_mask).count('1')
        
        if curr_size > best_size:
            best_size = curr_size
            best_mask = curr_mask
            
    # Iterated Local Search
    # We perturb the current best solution
    # Perturbation: remove k random vertices
    while time.time() - start_time < timeout:
        # Create a perturbed version of the best solution
        # Remove k random vertices
        k = random.randint(2, 8)
        curr_mask = best_mask
        
        # Get list of vertices in best_mask
        verts = [i for i in range(m) if (best_mask >> i) & 1]
        if len(verts) <= k:
            # If set is small, just restart from scratch
            order = list(range(m))
            random.shuffle(order)
            curr_mask = greedy_build(order)
        else:
            # Remove k random vertices
            to_remove = random.sample(verts, k)
            for v in to_remove:
                curr_mask &= ~(1 << v)
                
        # Re-optimize
        curr_mask = hill_climb(curr_mask)
        curr_size = bin(curr_mask).count('1')
        
        if curr_size > best_size:
            best_size = curr_size
            best_mask = curr_mask
            
        # Occasionally, do a full random restart to explore new areas
        if random.random() < 0.1:
            order = list(range(m))
            # Random order
            random.shuffle(order)
            # Or degree-based random
            # Let's mix: 50% random, 50% degree-noisy
            if random.random() < 0.5:
                random.shuffle(order)
            else:
                noisy_degrees = [d + random.uniform(0, 10) for d in degrees]
                order.sort(key=lambda x: noisy_degrees[x])
                
            new_mask = greedy_build(order)
            new_mask = hill_climb(new_mask)
            new_size = bin(new_mask).count('1')
            
            if new_size > best_size:
                best_size = new_size
                best_mask = new_mask

    result = np.array([i for i in range(m) if (best_mask >> i) & 1], dtype=np.int64)
    return result
