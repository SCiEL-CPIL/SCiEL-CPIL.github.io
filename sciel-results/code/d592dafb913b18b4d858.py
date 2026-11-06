import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict masks
    # conflict_mask[i] has bit j set if i and j conflict (i != j)
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Precompute conflict degree for each node
    conflict_degree = [bin(conflict_mask[i]).count('1') for i in range(N)]
    
    # 2-adic valuation
    def v2(n):
        c = 0
        while n % 2 == 0:
            c += 1
            n //= 2
        return c
    
    v2_val = [v2(i+1) for i in range(N)]
    
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    def get_size(mask):
        return bin(mask).count('1')
    
    def can_add(idx, mask):
        return (conflict_mask[idx] & mask) == 0
    
    def valid_set(mask):
        # Verify all pairs in mask are non-conflicting
        indices = [i for i in range(N) if (mask >> i) & 1]
        for ii in range(len(indices)):
            for jj in range(ii+1, len(indices)):
                a = indices[ii]
                b = indices[jj]
                if (conflict_mask[a] >> b) & 1:
                    return False
        return True
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy construction with different orderings
    # Sort by conflict degree (ascending - add low-conflict nodes first)
    sorted_by_degree = sorted(range(N), key=lambda i: conflict_degree[i])
    
    # Greedy: add nodes in order of increasing conflict degree
    mask = 0
    for i in sorted_by_degree:
        if can_add(i, mask):
            mask |= (1 << i)
    sz = get_size(mask)
    if sz > best_size:
        best_mask = mask
        best_size = sz
    
    # Greedy: start with all odds, then greedily add evens sorted by conflict with odds
    mask = 0
    for i in odds_indices:
        mask |= (1 << i)
    
    # Sort evens by number of conflicts with current mask
    def conflicts_with_mask(idx, m):
        return bin(conflict_mask[idx] & m).count('1')
    
    evens_sorted = sorted(evens_indices, key=lambda i: conflicts_with_mask(i, mask))
    for i in evens_sorted:
        if can_add(i, mask):
            mask |= (1 << i)
    sz = get_size(mask)
    if sz > best_size:
        best_mask = mask
        best_size = sz
    
    # Strategy 2: Simulated annealing / local search from best_mask
    current_mask = best_mask
    current_size = best_size
    
    def neighborhood_move(mask, size):
        """Try to improve the mask by removing k nodes and adding k+1 nodes."""
        indices_in = [i for i in range(N) if (mask >> i) & 1]
        indices_out = [i for i in range(N) if not ((mask >> i) & 1)]
        
        if not indices_in or not indices_out:
            return mask, size, False
        
        # Try removing 1, adding 2 (net +1)
        for _ in range(50):
            if time.time() > DEADLINE - 3:
                return mask, size, False
            to_remove = random.choice(indices_in)
            trial = mask & ~(1 << to_remove)
            
            # Try to add 2 non-conflicting nodes from outside
            candidates = [i for i in indices_out if can_add(i, trial)]
            if len(candidates) >= 2:
                # Pick 2 that don't conflict with each other
                for _ in range(20):
                    e1, e2 = random.sample(candidates, 2)
                    if not ((conflict_mask[e1] >> e2) & 1):
                        new_mask = trial | (1 << e1) | (1 << e2)
                        new_size = size + 1
                        return new_mask, new_size, True
        
        # Try removing 1, adding 1 (net 0, for exploration)
        for _ in range(30):
            if time.time() > DEADLINE - 3:
                return mask, size, False
            to_remove = random.choice(indices_in)
            trial = mask & ~(1 << to_remove)
            
            candidates = [i for i in indices_out if can_add(i, trial)]
            if candidates:
                e1 = random.choice(candidates)
                new_mask = trial | (1 << e1)
                return new_mask, size, True
        
        # Try removing 2, adding 3 (net +1)
        if len(indices_in) >= 2:
            for _ in range(20):
                if time.time() > DEADLINE - 3:
                    return mask, size, False
                to_remove1, to_remove2 = random.sample(indices_in, 2)
                trial = mask & ~(1 << to_remove1) & ~(1 << to_remove2)
                
                candidates = [i for i in indices_out if can_add(i, trial)]
                if len(candidates) >= 3:
                    for _ in range(10):
                        e1, e2, e3 = random.sample(candidates, 3)
                        if not ((conflict_mask[e1] >> e2) & 1) and \
                           not ((conflict_mask[e1] >> e3) & 1) and \
                           not ((conflict_mask[e2] >> e3) & 1):
                            new_mask = trial | (1 << e1) | (1 << e2) | (1 << e3)
                            new_size = size + 1
                            return new_mask, new_size, True
        
        return mask, size, False
    
    # Local search with simulated annealing
    temperature = 1.0
    cooling = 0.9995
    min_temp = 0.01
    
    for iteration in range(500000):
        if time.time() > DEADLINE - 2:
            break
        
        new_mask, new_size, made_move = neighborhood_move(current_mask, current_size)
        
        if made_move:
            if new_size >= current_size:
                # Accept if same or better size
                current_mask = new_mask
                current_size = new_size
                if new_size > best_size:
                    best_mask = new_mask
                    best_size = new_size
            else:
                # Accept worse moves with probability exp(-delta/T)
                delta = current_size - new_size
                if random.random() < math.exp(-delta / max(temperature, 1e-10)):
                    current_mask = new_mask
                    current_size = new_size
        
        temperature = max(temperature * cooling, min_temp)
        
        # Occasionally reheat
        if iteration % 10000 == 0 and temperature < 0.1:
            temperature = 0.5
    
    # Strategy 3: Multiple restarts with different initial sets
    for restart in range(20):
        if time.time() > DEADLINE - 5:
            break
        
        # Different starting points
        if restart % 4 == 0:
            # Start with all odds
            mask = 0
            for i in odds_indices:
                mask |= (1 << i)
        elif restart % 4 == 1:
            # Start with random subset of odds
            mask = 0
            for i in random.sample(odds_indices, len(odds_indices) // 2):
                mask |= (1 << i)
        elif restart % 4 == 2:
            # Start with low-conflict-degree nodes
            mask = 0
            for i in sorted_by_degree[:128]:
                if can_add(i, mask):
                    mask |= (1 << i)
        else:
            # Start with random
            mask = 0
            for i in random.sample(range(N), 100):
                if can_add(i, mask):
                    mask |= (1 << i)
        
        # Greedy add
        for i in sorted_by_degree:
            if not ((mask >> i) & 1) and can_add(i, mask):
                mask |= (1 << i)
        
        sz = get_size(mask)
        if sz > best_size:
            best_mask = mask
            best_size = sz
        
        # Short local search from this starting point
        current_mask = mask
        current_size = sz
        temperature = 0.5
        
        for iteration in range(50000):
            if time.time() > DEADLINE - 3:
                break
            
            new_mask, new_size, made_move = neighborhood_move(current_mask, current_size)
            
            if made_move:
                if new_size >= current_size:
                    current_mask = new_mask
                    current_size = new_size
                    if new_size > best_size:
                        best_mask = new_mask
                        best_size = new_size
                else:
                    delta = current_size - new_size
                    if random.random() < math.exp(-delta / max(temperature, 1e-10)):
                        current_mask = new_mask
                        current_size = new_size
            
            temperature = max(temperature * cooling, min_temp)
    
    # Final polish: try to add any remaining nodes
    for i in range(N):
        if not ((best_mask >> i) & 1) and can_add(i, best_mask):
            best_mask |= (1 << i)
    
    # Verify
    if not valid_set(best_mask):
        # Fallback to all odds
        best_mask = 0
        for i in odds_indices:
            best_mask |= (1 << i)
        best_size = len(odds_indices)
    
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    return result
