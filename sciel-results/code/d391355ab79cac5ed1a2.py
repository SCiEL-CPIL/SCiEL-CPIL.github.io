import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)

    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    # Start with all odds
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)
    
    best_mask = current_mask
    best_size = bin(current_mask).count('1')
    
    # Phase 1: Add evens with 0 conflicts
    changed = True
    while changed and time.time() < DEADLINE - 50:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 50:
                break
            if (current_mask >> e) & 1:
                continue
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
                best_mask = current_mask
                best_size = bin(current_mask).count('1')
                changed = True
    
    # Phase 2: Simulated Annealing with multi-element exchanges
    T = 2.0
    cooling = 0.9995
    max_iters = 200000
    
    current_size = best_size
    
    for i_iter in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        r = random.random()
        if r < 0.75:
            # Single add move: try to add a random element not in set
            if random.random() < 0.8:
                e_candidates = [idx for idx in evens_indices if not ((current_mask >> idx) & 1)]
                if not e_candidates:
                    continue
                e = random.choice(e_candidates)
            else:
                o_candidates = [idx for idx in odds_indices if not ((current_mask >> idx) & 1)]
                if not o_candidates:
                    continue
                e = random.choice(o_candidates)
            
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = bin(conflicts).count('1')
            new_size = current_size - num_conflicts + 1
            
            if new_size > current_size:
                current_mask = (current_mask & ~conflicts) | (1 << e)
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.5:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                        current_size = new_size
        elif r < 0.9:
            # Multi-element exchange: remove k elements, add 1
            k = random.choice([2, 3])
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if len(bits) < k:
                continue
            removed = random.sample(bits, k)
            
            # Find a candidate to add that doesn't conflict with remaining set
            # after removing 'removed'
            temp_mask = current_mask
            for rm in removed:
                temp_mask &= ~(1 << rm)
            
            # Candidates: elements not in temp_mask
            not_in_temp = [idx for idx in range(N) if not ((temp_mask >> idx) & 1)]
            if not not_in_temp:
                continue
            e = random.choice(not_in_temp)
            
            # Check if e conflicts with temp_mask (i.e., the set after removal)
            conflicts_with_temp = conflict_mask[e] & temp_mask
            if conflicts_with_temp != 0:
                continue
            
            # New size: current_size - k + 1
            new_size = current_size - k + 1
            
            if new_size > current_size:
                current_mask = temp_mask | (1 << e)
                current_size = new_size
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.3:
                        current_mask = temp_mask | (1 << e)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = temp_mask | (1 << e)
                        current_size = new_size
        else:
            # Random removal to escape local minima
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if bits:
                r_idx = random.choice(bits)
                current_mask &= ~(1 << r_idx)
                current_size -= 1
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final polish: greedy add any element with 0 conflicts
    changed = True
    while changed and time.time() < DEADLINE - 1:
        changed = False
        for idx in range(N):
            if time.time() > DEADLINE - 1:
                break
            if (best_mask >> idx) & 1:
                continue
            if (conflict_mask[idx] & best_mask) == 0:
                best_mask |= (1 << idx)
                best_size += 1
                changed = True

    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    
    return result
