import time
import random

def construct():
    deadline = time.time() + 85.0
    N = 256
    MAX_SIZE = 60
    
    def check_valid(vals):
        if not vals:
            return True
        n = len(vals)
        bits = [0] * (n + 1)
        bits[0] = 1
        
        for v in vals:
            for k in range(n - 1, -1, -1):
                if bits[k]:
                    bits[k + 1] |= (bits[k] << v)
        
        seen = 0
        for i in range(n + 1):
            if bits[i] & seen:
                return False
            seen |= bits[i]
        return True

    # Start with AP(31)
    # 226 to 256
    current_set = list(range(226, 257))
    
    # Verify start
    if not check_valid(current_set):
        # Fallback
        current_set = [1, 2, 4, 8, 16, 32, 64, 128]
        
    best_set = list(current_set)
    best_size = len(best_set)
    
    pool = set(range(1, N + 1))
    rng = random.Random(42)
    
    def try_add(val):
        nonlocal current_set
        if val in current_set:
            return False
        cand = current_set + [val]
        if check_valid(cand):
            current_set = cand
            return True
        return False

    def try_swap(val_out, val_in):
        nonlocal current_set
        if val_in in current_set:
            return False
        cand = [x for x in current_set if x != val_out]
        cand.append(val_in)
        if check_valid(cand):
            current_set = cand
            return True
        return False

    # Iterative improvement
    while time.time() < deadline - 5:
        # Try to add
        added = False
        # Try all values in pool not in set
        # Sort by some heuristic? Maybe random order to explore more
        candidates = list(pool - set(current_set))
        rng.shuffle(candidates)
        
        for v in candidates:
            if try_add(v):
                added = True
                best_size = len(current_set)
                if len(current_set) > len(best_set):
                    best_set = list(current_set)
                break
        
        if added:
            continue
            
        # If no add, try to swap to find a better configuration
        # that might allow an add
        # Perform a limited number of random swaps
        swap_attempts = 50
        for _ in range(swap_attempts):
            if time.time() > deadline - 2:
                break
            
            idx = rng.randint(0, len(current_set) - 1)
            val_out = current_set[idx]
            
            candidates = list(pool - set(current_set))
            if not candidates:
                break
            val_in = rng.choice(candidates)
            
            if try_swap(val_out, val_in):
                # After a successful swap, try to add again immediately
                candidates = list(pool - set(current_set))
                rng.shuffle(candidates)
                for v in candidates:
                    if try_add(v):
                        best_size = len(current_set)
                        if len(current_set) > len(best_set):
                            best_set = list(current_set)
                        break
                # If we added, the outer loop will continue and try to add more
                # If we didn't add, we continue swapping

    # Ensure best_set is the largest found
    if len(current_set) > len(best_set):
        best_set = list(current_set)
        
    return [int(x) for x in best_set]
