import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # seconds, leave 10s margin
    
    n = 16
    k = 4
    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for combo in combinations(range(1, n+1), k):
        mask = 0
        for x in combo:
            mask |= (1 << (x-1))
        all_subsets.append((combo, mask))
    
    total = len(all_subsets)  # 1820
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def is_sunflower_core(mask_a, mask_b, mask_c):
        """Check if three sets form a sunflower: all pairwise intersections equal."""
        ab = mask_a & mask_b
        ac = mask_a & mask_c
        bc = mask_b & mask_c
        return ab == ac == bc
    
    def check_valid_greedy(chosen_masks, new_mask):
        """Check if adding new_mask to chosen_masks creates a sunflower triple."""
        for i in range(len(chosen_masks)):
            for j in range(i+1, len(chosen_masks)):
                if is_sunflower_core(chosen_masks[i], chosen_masks[j], new_mask):
                    return False
        return True
    
    def greedy_build(order):
        """Build a family greedily following the given order."""
        chosen = []
        for idx in order:
            if time_left() < 5.0:
                break
            mask = all_subsets[idx][1]
            valid = True
            for i in range(len(chosen)):
                for j in range(i+1, len(chosen)):
                    if is_sunflower_core(chosen[i], chosen[j], mask):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                chosen.append(mask)
        return chosen
    
    def local_search(chosen, iterations=500):
        """Try to improve the family via local search."""
        chosen_set = set(chosen)
        best = list(chosen)
        
        for _ in range(iterations):
            if time_left() < 5.0:
                break
            
            # Try adding a random unchosen set
            added = False
            for _ in range(20):
                if time_left() < 5.0:
                    break
                idx = random.randint(0, total-1)
                mask = all_subsets[idx][1]
                if mask in chosen_set:
                    continue
                # Check validity
                valid = True
                for i in range(len(chosen_set)):
                    for j in range(i+1, len(chosen_set)):
                        if is_sunflower_core(list(chosen_set)[i], list(chosen_set)[j], mask):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    chosen_set.add(mask)
                    added = True
                    break
            
            # Try swapping: remove one, add another
            if not added and len(chosen_set) > 0:
                for _ in range(10):
                    if time_left() < 5.0:
                        break
                    to_remove = random.choice(list(chosen_set))
                    chosen_set.remove(to_remove)
                    for _ in range(5):
                        idx = random.randint(0, total-1)
                        mask = all_subsets[idx][1]
                        if mask in chosen_set:
                            continue
                        valid = True
                        for i in range(len(chosen_set)):
                            for j in range(i+1, len(chosen_set)):
                                if is_sunflower_core(list(chosen_set)[i], list(chosen_set)[j], mask):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            chosen_set.add(mask)
                            break
                    if len(chosen_set) > len(best):
                        best = list(chosen_set)
        
        return best
    
    best_family = []
    
    # Multiple random greedy builds
    num_restarts = 30
    for restart in range(num_restarts):
        if time_left() < 10.0:
            break
        
        order = list(range(total))
        random.shuffle(order)
        
        # Try different heuristics for ordering
        if restart % 3 == 0:
            # Random order
            pass
        elif restart % 3 == 1:
            # Sort by some property
            random.shuffle(order)
        else:
            # Reverse random
            random.shuffle(order)
            order.reverse()
        
        family = greedy_build(order)
        
        if len(family) > len(best_family):
            best_family = family
        
        # Local search on the best found so far
        if len(family) >= len(best_family) - 2 and time_left() > 15.0:
            improved = local_search(family, iterations=200)
            if len(improved) > len(best_family):
                best_family = improved
    
    # Final local search on the best
    if time_left() > 10.0:
        improved = local_search(best_family, iterations=300)
        if len(improved) > len(best_family):
            best_family = improved
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(n):
            if mask & (1 << i):
                elements.append(i+1)
        result.append(sorted(elements))
    
    # Ensure we don't exceed 250
    if len(result) > 250:
        result = result[:250]
    
    return result

construct()
