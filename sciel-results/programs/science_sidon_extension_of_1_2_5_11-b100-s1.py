"""Search-based solver for Sidon set extension containing {1, 2, 5, 11} in [1, 284]."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s budget, 20s safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    # Pre-compute initial sums
    initial_sums = set()
    for i, x in enumerate(prescribed):
        for y in prescribed[i:]:
            initial_sums.add(x + y)
    
    # Best found so far
    best_set = set(prescribed)
    best_size = len(best_set)
    best_sums = initial_sums
    
    def build_sums(s):
        """Build the set of all pairwise sums a+b for a<=b in s."""
        sl = sorted(s)
        sums = set()
        for i, x in enumerate(sl):
            for y in sl[i:]:
                sums.add(x + y)
        return sums

    # Helper to check if adding x to a set with given sums is valid
    def can_add(x, current_set, current_sums):
        new_sums = {x + y for y in current_set} | {2 * x}
        if len(new_sums) == len(current_set) + 1 and not (new_sums & current_sums):
            return True
        return False

    def get_new_sums(x, current_set, current_sums):
        return {x + y for y in current_set} | {2 * x}

    # Strategy 1: Greedy forward construction with randomization
    remaining = [x for x in range(12, 285) if x not in prescribed]
    
    def greedy_with_random_order(order, time_limit=5.0):
        nonlocal best_set, best_size, best_sums
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        start = time.time()
        
        for x in order:
            if time.time() - start > time_limit:
                break
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
        
        return len(current_set)

    # Try a few quick greedy passes with different random seeds
    rng = random.Random(42)
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        order = remaining[:]
        rng.shuffle(order)
        greedy_with_random_order(order, time_limit=0.5)
    
    # Strategy 2: Fast local search with incremental sum maintenance
    def fast_local_search(start_set, time_limit=60.0):
        nonlocal best_set, best_size, best_sums
        current_set = set(start_set)
        current_sums = build_sums(current_set)
        start = time.time()
        
        # Precompute available candidates
        all_candidates = list(range(1, 285))
        
        for _ in range(200000):
            if time.time() - start > time_limit or time.time() > deadline - 5:
                break
            
            # Try to add a random element
            if len(current_set) < 284:
                # Pick random candidate not in set
                for _attempt in range(50):
                    x = rng.choice(all_candidates)
                    if x not in current_set:
                        if can_add(x, current_set, current_sums):
                            new_sums = get_new_sums(x, current_set, current_sums)
                            current_set.add(x)
                            current_sums.update(new_sums)
                            if len(current_set) > best_size:
                                best_set = set(current_set)
                                best_size = len(current_set)
                                best_sums = set(current_sums)
                        break
            
            # Try to swap: remove one non-prescribed, add one new
            removable = [x for x in current_set if x not in prescribed_set]
            if removable:
                x_remove = rng.choice(removable)
                # Compute sums without x_remove efficiently
                # sums to remove: all sums involving x_remove
                sums_to_remove = {x_remove + y for y in current_set} | {2 * x_remove}
                # But we need sums of remaining set
                remaining_set = current_set - {x_remove}
                # New sums without x_remove: current_sums minus sums involving x_remove
                # But careful: some sums might be shared. Let's rebuild for correctness
                # Actually, let's just try to find a valid x_add for the reduced set
                reduced_sums = build_sums(remaining_set)
                
                for _attempt in range(30):
                    x_add = rng.choice(all_candidates)
                    if x_add in current_set:
                        continue
                    # Check if x_add can be added to remaining_set
                    new_sums_test = {x_add + y for y in remaining_set} | {2 * x_add}
                    if len(new_sums_test) == len(remaining_set) + 1 and not (new_sums_test & reduced_sums):
                        # Valid swap
                        current_set = remaining_set | {x_add}
                        current_sums = reduced_sums | new_sums_test
                        if len(current_set) > best_size:
                            best_set = set(current_set)
                            best_size = len(current_set)
                            best_sums = set(current_sums)
                        break
        
        return len(current_set)

    # Run fast local search from best found
    if best_size > len(prescribed):
        fast_local_search(best_set, time_limit=40.0)
    
    # Strategy 3: Systematic greedy strategies
    # Greedy that picks the smallest valid element each time
    if time.time() < deadline - 15:
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        for x in range(12, 285):
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
    
    # Greedy that picks the largest valid element each time
    if time.time() < deadline - 15:
        current_set = set(prescribed)
        current_sums = set(initial_sums)
        for x in range(284, 11, -1):
            if x in current_set:
                continue
            if can_add(x, current_set, current_sums):
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            best_sums = set(current_sums)
    
    # Strategy 4: Greedy with lookahead - try multiple candidates at each step
    if time.time() < deadline - 15:
        def greedy_with_lookahead(time_limit=5.0):
            nonlocal best_set, best_size, best_sums
            current_set = set(prescribed)
            current_sums = set(initial_sums)
            start = time.time()
            
            while time.time() - start < time_limit:
                # Find all valid additions
                valid_adds = []
                for x in range(1, 285):
                    if x not in current_set and can_add(x, current_set, current_sums):
                        valid_adds.append(x)
                
                if not valid_adds:
                    break
                
                # Pick randomly among valid additions
                x = rng.choice(valid_adds)
                new_sums = get_new_sums(x, current_set, current_sums)
                current_set.add(x)
                current_sums.update(new_sums)
            
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
                best_sums = set(current_sums)
            return len(current_set)
        
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            greedy_with_lookahead(time_limit=2.0)
    
    # More randomized greedy passes
    if time.time() < deadline - 10:
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            order = remaining[:]
            rng.shuffle(order)
            greedy_with_random_order(order, time_limit=1.0)
    
    # Final fast local search with more time
    if time.time() < deadline - 5 and best_size > len(prescribed):
        fast_local_search(best_set, time_limit=30.0)
    
    # Final check: ensure we return a valid set
    result = sorted(best_set)
    
    # Verify validity
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for y in result[i:]:
            s = x + y
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    
    if not valid:
        # Fall back to prescribed only
        return prescribed
    
    return result

# For testing
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result}")
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for y in result[i:]:
            s = x + y
            if s in sums:
                valid = False
                break
            sums.add(s)
    print(f"Valid: {valid}")
