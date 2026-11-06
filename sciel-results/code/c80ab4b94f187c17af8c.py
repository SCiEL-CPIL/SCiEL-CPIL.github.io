import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    N = 256
    
    # Precompute conflict masks
    # conflicts[i] is a bitmask of j (1-indexed positions) such that (i*j) % (i+j) == 0
    conflicts = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Full mask of all vertices 1..N
    full_mask = 0
    for i in range(1, N + 1):
        full_mask |= (1 << i)
    
    def popcount(x):
        return x.bit_count()
    
    def mask_to_set(mask):
        return [i for i in range(1, N + 1) if (mask >> i) & 1]
    
    # Greedy minimum-degree for initial solution
    def greedy_min_degree():
        current_available = full_mask
        solution_mask = 0
        
        while current_available:
            # Find node with min degree in current_available
            best_node = -1
            min_deg = N + 1
            
            temp = current_available
            while temp:
                lsb = temp & -temp
                x = lsb.bit_length() - 1
                deg = popcount(conflicts[x] & current_available)
                if deg < min_deg:
                    min_deg = deg
                    best_node = x
                temp -= lsb
                
            if best_node == -1:
                break
            
            solution_mask |= (1 << best_node)
            remove_mask = (1 << best_node) | conflicts[best_node]
            current_available &= ~remove_mask
            
        return solution_mask
    
    # Greedy random
    def greedy_random(seed=None):
        if seed is not None:
            random.seed(seed)
        solution_mask = 0
        nodes = list(range(1, N + 1))
        random.shuffle(nodes)
        
        for x in nodes:
            if (conflicts[x] & solution_mask) == 0:
                solution_mask |= (1 << x)
                
        return solution_mask
    
    # Get initial solution
    best_mask = greedy_min_degree()
    best_size = popcount(best_mask)
    
    # Try a few random greeds to get a better initial bound
    for _ in range(5):
        m = greedy_random()
        s = popcount(m)
        if s > best_size:
            best_size = s
            best_mask = m
    
    # Tomita's algorithm for Maximum Independent Set
    # candidates: bitmask of vertices still available
    # current: bitmask of vertices in current independent set
    # current_size: number of vertices in current
    # We use recursion with branching
    
    def branch_and_bound(candidates, current_mask, current_size):
        nonlocal best_mask, best_size
        
        if time.time() > deadline - 1:
            return
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Pruning: even if we add all candidates, can't beat best
        if current_size + popcount(candidates) <= best_size:
            return
        
        # Select vertex to branch on
        # Heuristic: pick vertex with highest degree in candidates subgraph
        # This tends to reduce the candidate set faster
        v = -1
        max_deg = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            x = lsb.bit_length() - 1
            deg = popcount(conflicts[x] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = x
            temp -= lsb
        
        if v == -1:
            return
        
        # Branch 1: Include v
        # New candidates = candidates - {v} - neighbors(v)
        new_candidates = candidates & ~(1 << v) & ~conflicts[v]
        branch_and_bound(new_candidates, current_mask | (1 << v), current_size + 1)
        
        # Branch 2: Exclude v
        # New candidates = candidates - {v}
        new_candidates = candidates & ~(1 << v)
        branch_and_bound(new_candidates, current_mask, current_size)
    
    # Run B&B with time limit
    branch_and_bound(full_mask, 0, 0)
    
    # If we still have time, try to improve with local search
    if time.time() < deadline - 5:
        # Local search: try swaps
        for _ in range(100):
            if time.time() > deadline - 2:
                break
            
            mask = best_mask
            improved = False
            nodes_in_set = mask_to_set(mask)
            random.shuffle(nodes_in_set)
            
            for x in nodes_in_set:
                if time.time() > deadline - 2:
                    break
                    
                new_mask = mask & ~(1 << x)
                
                # Find compatible candidates not in new_mask
                candidates = []
                for y in range(1, N + 1):
                    if not (new_mask >> y) & 1:
                        if (conflicts[y] & new_mask) == 0:
                            candidates.append(y)
                            
                if len(candidates) >= 2:
                    random.shuffle(candidates)
                    trial_mask = new_mask
                    added = 0
                    for y in candidates:
                        if (conflicts[y] & trial_mask) == 0:
                            trial_mask |= (1 << y)
                            added += 1
                            
                    if added > 1:
                        mask = trial_mask
                        improved = True
                        break
                        
            if improved:
                s = popcount(mask)
                if s > best_size:
                    best_size = s
                    best_mask = mask
            else:
                # Try a fresh greedy + B&B on remaining time
                if time.time() > deadline - 10:
                    break
                m = greedy_random()
                s = popcount(m)
                if s > best_size:
                    best_size = s
                    best_mask = m
                    # Quick B&B from here
                    # We'd need to restart B&B, but that's complex.
                    # Just use the greedy result.
    
    return mask_to_set(best_mask)
