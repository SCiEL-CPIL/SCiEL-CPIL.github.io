"""Direct construction of maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import random

# LCM of 1..22
_LCM = 232792560

def _verify(subset):
    """Check if all 2^k subset sums of {LCM//x for x in subset} are distinct."""
    k = len(subset)
    if k == 0:
        return True
    scaled = [_LCM // x for x in subset]
    # Generate all subset sums
    n = 1 << k
    sums = [0] * n
    for mask in range(1, n):
        lb = mask & (-mask)
        idx = lb.bit_length() - 1
        sums[mask] = sums[mask ^ lb] + scaled[idx]
    return len(set(sums)) == n

def _verify_fast(subset):
    """Faster verification using numpy for larger sets."""
    k = len(subset)
    if k == 0:
        return True
    if k <= 16:
        return _verify(subset)
    # For larger sets, use a hash-based approach with numpy
    scaled = [_LCM // x for x in subset]
    n = 1 << k
    # Use a set of sums but generate more efficiently
    sums = [0] * n
    for i in range(k):
        for mask in range(1 << i, 1 << (i + 1)):
            sums[mask] = sums[mask ^ (1 << i)] + scaled[i]
    return len(set(sums)) == n

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    universe = list(range(1, 23))
    
    # Phase 1: Greedy construction (same as before)
    order = sorted(universe, key=lambda x: -(_LCM // x))
    
    current = []
    for x in order:
        if time.time() > deadline - 30:
            break
        candidate = current + [x]
        if _verify(candidate):
            current = candidate
    
    best = list(current)
    best_size = len(best)
    
    # Try adding remaining elements
    in_set = set(best)
    remaining = [x for x in universe if x not in in_set]
    for x in remaining:
        if time.time() > deadline - 30:
            break
        candidate = best + [x]
        if _verify(candidate):
            best = candidate
            in_set = set(best)
            best_size = len(best)
    
    # Phase 2: Simulated annealing local search
    # Objective: maximize cardinality while maintaining dissociation
    # Moves: swap one element in/out, or add/remove an element
    
    rng = random.Random(42)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.995
    
    current_set = list(best)
    current_size = len(current_set)
    T = T_start
    
    max_iters = 50000
    iter_count = 0
    
    while iter_count < max_iters and time.time() < deadline - 10:
        iter_count += 1
        
        if time.time() > deadline - 10:
            break
        
        # Choose a move
        in_set = set(current_set)
        not_in_set = [x for x in universe if x not in in_set]
        
        move_type = rng.random()
        
        if move_type < 0.4 and not_in_set:
            # Add a random element
            x = rng.choice(not_in_set)
            new_set = current_set + [x]
            new_size = len(new_set)
            
            if _verify(new_set):
                # Accept if it increases size, or with probability based on temperature
                if new_size > current_size:
                    current_set = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best = list(current_set)
                        best_size = current_size
                elif T > T_end:
                    # Accept worse (same size, still valid) with probability exp(-1/T)
                    if rng.random() < 0.1:  # Small chance to explore
                        current_set = new_set
                        current_size = new_size
            else:
                # Invalid move - reject
                pass
                
        elif move_type < 0.7 and len(current_set) > 1:
            # Remove a random element
            idx = rng.randint(0, len(current_set) - 1)
            new_set = current_set[:idx] + current_set[idx+1:]
            new_size = len(new_set)
            
            # Only accept if we can potentially add more later
            # For now, only accept if size stays same or we're exploring
            if new_size >= current_size - 1 and T > 0.5:
                if rng.random() < 0.3:  # Small chance to remove for exploration
                    current_set = new_set
                    current_size = new_size
                    
        elif move_type < 0.95 and not_in_set and len(current_set) > 0:
            # Swap: remove one, add one
            idx = rng.randint(0, len(current_set) - 1)
            x_add = rng.choice(not_in_set)
            x_remove = current_set[idx]
            
            new_set = current_set[:idx] + current_set[idx+1:] + [x_add]
            new_size = len(new_set)
            
            if _verify(new_set):
                if new_size > current_size:
                    current_set = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best = list(current_set)
                        best_size = current_size
                elif T > T_end and rng.random() < 0.2:
                    # Accept same-size swap for exploration
                    current_set = new_set
                    current_size = new_size
            else:
                pass
        else:
            # Try to add multiple elements at once
            if not_in_set:
                x1 = rng.choice(not_in_set)
                new_set = current_set + [x1]
                if _verify(new_set) and len(new_set) > current_size:
                    current_set = new_set
                    current_size = len(new_set)
                    if current_size > best_size:
                        best = list(current_set)
                        best_size = current_size
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    # Phase 3: Final greedy pass on the best set found
    in_set = set(best)
    remaining = [x for x in universe if x not in in_set]
    for x in remaining:
        if time.time() > deadline - 3:
            break
        candidate = best + [x]
        if _verify(candidate):
            best = candidate
    
    # Phase 4: Try swaps to improve the final set
    if time.time() < deadline - 5 and len(best) < 22:
        in_set = set(best)
        not_in_set = [x for x in universe if x not in in_set]
        
        for i, a in enumerate(best):
            if time.time() > deadline - 3:
                break
            for b in not_in_set:
                candidate = best[:]
                candidate[i] = b
                if _verify(candidate):
                    new_in = set(candidate)
                    new_not = [x for x in universe if x not in new_in]
                    extra = 0
                    for c in new_not:
                        if time.time() > deadline - 3:
                            break
                        if _verify(candidate + [c]):
                            extra += 1
                    if extra > 0:
                        best = candidate
                        in_set = set(best)
                        not_in_set = [x for x in universe if x not in in_set]
                        break
    
    # Final verification
    if not _verify(best):
        # Fallback to a known valid set
        return [1, 2, 4, 8, 16]
    
    return best
