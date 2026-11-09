import time
import random

# LCM of 1..22
_LCM = 232792560

def _is_valid(subset):
    """Check if all subset sums of reciprocals are distinct (via LCM scaling)."""
    if not subset:
        return True
    weights = [_LCM // i for i in subset]
    sums = {0}
    for w in weights:
        new_sums = {s + w for s in sums}
        if new_sums & sums:
            return False
        sums |= new_sums
    return True

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 60s. Use 55s to be safe.
    deadline = start_time + 55.0

    # Known valid seed (size 9)
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_subset)
    all_elems = list(range(1, 23))
    
    def time_ok():
        return time.time() < deadline

    # Strategy: Simulated Annealing / Local Search
    # 1. Greedy expansion from seed
    # 2. Randomized Local Search with Swaps (Annealing)
    
    # Initial Greedy Expansion
    current = list(best_subset)
    current_set = set(current)
    
    # Try to add elements greedily
    improved = True
    while improved and time_ok():
        improved = False
        for c in all_elems:
            if c in current_set:
                continue
            if _is_valid(current + [c]):
                current.append(c)
                current_set.add(c)
                improved = True
                if len(current) > best_size:
                    best_size = len(current)
                    best_subset = list(current)
                break # Restart loop as set changed

    # Local Search / Annealing Loop
    # We perform rounds of local search. In each round, we try to add elements.
    # If we can't add, we perform swaps to explore the landscape.
    
    for _ in range(1000): # Max 1000 rounds, time_ok will cut short
        if not time_ok():
            break
        
        # Occasionally restart from a random greedy build to escape local minima
        if random.random() < 0.2:
            candidates = list(all_elems)
            random.shuffle(candidates)
            current = []
            current_set = set()
            for c in candidates:
                if not time_ok(): break
                if _is_valid(current + [c]):
                    current.append(c)
                    current_set.add(c)
            if len(current) > best_size:
                best_size = len(current)
                best_subset = list(current)
        else:
            # Start from the current best known valid set
            current = list(best_subset)
            current_set = set(current)

        # Inner Local Search Loop
        local_improved = True
        while local_improved and time_ok():
            local_improved = False
            
            # 1. Try to add any element (Expansion)
            for c in all_elems:
                if c in current_set:
                    continue
                if _is_valid(current + [c]):
                    current.append(c)
                    current_set.add(c)
                    local_improved = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_subset = list(current)
                    break # Restart inner loop after successful add

            # 2. If no add worked, try a swap (Annealing move)
            if not local_improved and len(current) > 0:
                # Try a few random swaps to see if we can find a better configuration
                # that might allow future additions
                for _ in range(10):
                    if not time_ok(): break
                    if not current: break
                    
                    idx_remove = random.randrange(len(current))
                    c_remove = current[idx_remove]
                    
                    # Pick a candidate to add
                    candidates_to_add = [c for c in all_elems if c not in current_set]
                    if not candidates_to_add:
                        break
                    c_add = random.choice(candidates_to_add)
                    
                    # Perform swap
                    new_current = current[:idx_remove] + current[idx_remove+1:] + [c_add]
                    new_current_set = set(new_current)
                    
                    if _is_valid(new_current):
                        # Accept the swap (Neutral move in SA)
                        current = new_current
                        current_set = new_current_set
                        local_improved = True
                        break # Restart inner loop after swap

    return best_subset
