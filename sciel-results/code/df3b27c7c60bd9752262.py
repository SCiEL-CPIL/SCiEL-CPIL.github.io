import time
import random

N = 512
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Precompute all 4-term arithmetic progressions in [1, 512]
    # For bitmask-based conflict checking, we store for each element x,
    # a list of masks representing the other 3 elements in each 4-AP containing x
    aps_masks = {i: [] for i in range(1, N + 1)}
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # The AP is (a, b, c, e)
            # For each element, store a bitmask of the other three
            mask_bce = (1 << b) | (1 << c) | (1 << e)
            mask_ace = (1 << a) | (1 << c) | (1 << e)
            mask_abe = (1 << a) | (1 << b) | (1 << e)
            mask_abc = (1 << a) | (1 << b) | (1 << c)
            
            aps_masks[a].append(mask_bce)
            aps_masks[b].append(mask_ace)
            aps_masks[c].append(mask_abe)
            aps_masks[e].append(mask_abc)

    def can_add_bitmask(x, current_mask):
        """Check if adding x to current_mask creates a 4-term AP using bitmask operations"""
        for mask in aps_masks[x]:
            if (current_mask & mask) == mask:
                return False
        return True

    def greedy_randomized_bitmask():
        """Greedy with random ordering using bitmask"""
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        current_mask = 0
        for x in elements:
            if can_add_bitmask(x, current_mask):
                current_mask |= (1 << x)
        return current_mask

    def local_search_bitmask(current_mask):
        """
        Local search using bitmask:
        1. Try to add any valid element (greedy fill).
        2. If stuck, try swapping: remove a random element, try to add multiple new ones.
        """
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT:
            improved = False
            
            # Phase 1: Greedy fill
            added_any = True
            while added_any and time.time() - start_time < TIME_LIMIT:
                added_any = False
                # Try to find any element that can be added
                candidates = [x for x in range(1, N + 1) if not (current_mask & (1 << x))]
                random.shuffle(candidates)
                for x in candidates:
                    if can_add_bitmask(x, current_mask):
                        current_mask |= (1 << x)
                        added_any = True
                        break

            # Phase 2: Swap
            if not added_any:
                # Pick a random element to remove
                if current_mask:
                    # Get list of elements in current set
                    current_elements = [x for x in range(1, N + 1) if (current_mask & (1 << x))]
                    if current_elements:
                        remove_x = random.choice(current_elements)
                        trial_mask = current_mask & ~(1 << remove_x)
                        # Try to fill with new elements
                        candidates = [x for x in range(1, N + 1) if not (trial_mask & (1 << x))]
                        random.shuffle(candidates)
                        for x in candidates:
                            if can_add_bitmask(x, trial_mask):
                                trial_mask |= (1 << x)
                        if bin(trial_mask).count('1') > bin(current_mask).count('1'):
                            current_mask = trial_mask
                            improved = True
                            break

        return current_mask

    best_mask = 0
    best_size = 0
    
    # Run multiple restarts
    for i in range(500):
        if time.time() - start_time > TIME_LIMIT - 3:
            break
        
        current_mask = greedy_randomized_bitmask()
        current_mask = local_search_bitmask(current_mask)
        
        current_size = bin(current_mask).count('1')
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
    
    # Try base-3 starting point
    if time.time() - start_time < TIME_LIMIT - 2:
        base3_mask = 0
        for i in range(1, N + 1):
            n = i
            valid = True
            while n > 0:
                if n % 3 == 2:
                    valid = False
                    break
                n //= 3
            if valid:
                base3_mask |= (1 << i)
        
        if bin(base3_mask).count('1') > best_size:
            best_mask = base3_mask
            best_size = bin(base3_mask).count('1')
        
        improved_mask = local_search_bitmask(best_mask)
        if bin(improved_mask).count('1') > best_size:
            best_mask = improved_mask
            best_size = bin(improved_mask).count('1')
    
    # Convert best_mask back to sorted list
    best_set = [x for x in range(1, N + 1) if (best_mask & (1 << x))]
    return best_set
