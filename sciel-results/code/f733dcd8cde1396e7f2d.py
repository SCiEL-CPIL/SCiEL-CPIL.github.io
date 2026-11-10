import time
import random
import itertools

def check_validity(S):
    """
    Returns True if S is a valid set.
    S is a list of distinct ints.
    """
    if len(S) == 0:
        return True
    sums = set()
    for i in range(1, len(S) + 1):
        for comb in itertools.combinations(S, i):
            s = sum(comb)
            if s in sums:
                return False
            sums.add(s)
    
    sums_list = list(sums)
    for i in range(len(sums_list)):
        u = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            v = sums_list[j]
            if u > v:
                u, v = v, u
            if v % u == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = []
    best_size = 0
    
    # Precompute: for each x in 1..256, count how many multiples of x are in 1..256
    # Higher count means more potential divisibility conflicts
    multiple_counts = {}
    for x in range(1, 257):
        count = 0
        for m in range(2*x, 257, x):
            count += 1
        multiple_counts[x] = count
    
    for _ in range(20000):
        if time.time() > deadline:
            break
            
        # Randomize candidate order, but bias towards numbers with fewer multiples
        # (fewer potential divisibility conflicts)
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        # Sort by multiple count (ascending) with some randomness to maintain diversity
        # Use a weighted approach: shuffle, then partially sort
        for i in range(len(candidates) - 1, 0, -1):
            if random.random() < 0.3:  # 30% chance to swap to a "better" position
                # Find the candidate with minimum multiple count in [0..i]
                min_idx = i
                for j in range(i):
                    if multiple_counts[candidates[j]] < multiple_counts[candidates[min_idx]]:
                        min_idx = j
                if min_idx != i:
                    candidates[i], candidates[min_idx] = candidates[min_idx], candidates[i]
        
        current_set = []
        current_sums = set()  # Maintain subset sums incrementally
        
        # Greedy phase with incremental sum tracking
        for x in candidates:
            # New sums when adding x: {x} union {s + x for s in current_sums}
            new_sums = {x}
            for s in current_sums:
                new_sums.add(s + x)
            
            # Check for duplicates: new_sums must be disjoint from current_sums
            if new_sums & current_sums:
                continue
            
            # Check divisibility between new_sums and current_sums
            valid = True
            for u in new_sums:
                for v in current_sums:
                    if u == v:
                        continue
                    if u < v:
                        if v % u == 0:
                            valid = False
                            break
                    else:
                        if u % v == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check divisibility within new_sums
            new_list = list(new_sums)
            for i in range(len(new_list)):
                u = new_list[i]
                for j in range(i + 1, len(new_list)):
                    v = new_list[j]
                    if u < v:
                        if v % u == 0:
                            valid = False
                            break
                    else:
                        if u % v == 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                current_set.append(x)
                current_sums = current_sums | new_sums
        
        # Local Search Phase: try to add more elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            available = [x for x in range(1, 257) if x not in current_set]
            if not available:
                break
            random.shuffle(available)
            for x in available[:20]:
                # New sums when adding x
                new_sums = {x}
                for s in current_sums:
                    new_sums.add(s + x)
                
                if new_sums & current_sums:
                    continue
                
                valid = True
                for u in new_sums:
                    for v in current_sums:
                        if u == v:
                            continue
                        if u < v:
                            if v % u == 0:
                                valid = False
                                break
                        else:
                            if u % v == 0:
                                valid = False
                                break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                new_list = list(new_sums)
                for i in range(len(new_list)):
                    u = new_list[i]
                    for j in range(i + 1, len(new_list)):
                        v = new_list[j]
                        if u < v:
                            if v % u == 0:
                                valid = False
                                break
                        else:
                            if u % v == 0:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    current_set.append(x)
                    current_sums = current_sums | new_sums
                    improved = True
                    break
        
        # Try swaps to enable further additions
        if time.time() < deadline and len(current_set) > 0:
            available = [x for x in range(1, 257) if x not in current_set]
            if available:
                # Try a few random swaps
                for _swap_attempt in range(5):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, len(current_set) - 1)
                    old_x = current_set[idx]
                    new_x = random.choice(available)
                    
                    # Build new set
                    test_set = current_set[:]
                    test_set[idx] = new_x
                    
                    # Recompute sums for test_set
                    test_sums = set()
                    for i in range(1, len(test_set) + 1):
                        for comb in itertools.combinations(test_set, i):
                            s = sum(comb)
                            if s in test_sums:
                                break
                            test_sums.add(s)
                        else:
                            continue
                        break
                    else:
                        # No duplicate sums, check divisibility
                        test_sums_list = list(test_sums)
                        div_valid = True
                        for i in range(len(test_sums_list)):
                            u = test_sums_list[i]
                            for j in range(i + 1, len(test_sums_list)):
                                v = test_sums_list[j]
                                if u > v:
                                    u, v = v, u
                                if v % u == 0:
                                    div_valid = False
                                    break
                            if not div_valid:
                                break
                        
                        if div_valid:
                            # Check if this swap allows adding more
                            new_available = [x for x in range(1, 257) if x not in test_set]
                            random.shuffle(new_available)
                            for x in new_available[:10]:
                                new_sums = {x}
                                for s in test_sums:
                                    new_sums.add(s + x)
                                if new_sums & test_sums:
                                    continue
                                valid_add = True
                                for u in new_sums:
                                    for v in test_sums:
                                        if u == v:
                                            continue
                                        if u < v:
                                            if v % u == 0:
                                                valid_add = False
                                                break
                                        else:
                                            if u % v == 0:
                                                valid_add = False
                                                break
                                    if not valid_add:
                                        break
                                if valid_add:
                                    nl = list(new_sums)
                                    for i in range(len(nl)):
                                        u = nl[i]
                                        for j in range(i + 1, len(nl)):
                                            v = nl[j]
                                            if u < v:
                                                if v % u == 0:
                                                    valid_add = False
                                                    break
                                            else:
                                                if u % v == 0:
                                                    valid_add = False
                                                    break
                                        if not valid_add:
                                            break
                                if valid_add:
                                    current_set = test_set
                                    current_sums = test_sums
                                    current_set.append(x)
                                    current_sums = current_sums | new_sums
                                    break
                            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return best_set

def _is_valid_addition(S, x):
    """
    Check if adding x to S keeps the set valid.
    S is a list of ints (current set).
    x is the new int.
    """
    # Compute existing sums
    sums = set()
    for i in range(1, len(S) + 1):
        for comb in itertools.combinations(S, i):
            sums.add(sum(comb))
            
    # New sums
    new_sums = set()
    new_sums.add(x)
    for s in sums:
        new_sums.add(s + x)
        
    # Check for duplicates (sums & new_sums)
    if sums & new_sums:
        return False
        
    all_sums = sums | new_sums
    
    # Check divisibility
    # New sums are x and s+x.
    # Old sums are s.
    # We need to check:
    # 1. u in new_sums, v in sums: u|v or v|u
    # 2. u in new_sums, v in new_sums: u|v or v|u (u!=v)
    
    # Optimization: Only check pairs involving at least one new sum.
    # Since S was valid, sums are pairwise non-dividing.
    
    # Check new vs old
    for u in new_sums:
        for v in sums:
            if u == v: continue # Duplicate check handled above, but just in case
            if u < v:
                if v % u == 0:
                    return False
            else:
                if u % v == 0:
                    return False
                    
    # Check new vs new
    new_list = list(new_sums)
    for i in range(len(new_list)):
        u = new_list[i]
        for j in range(i+1, len(new_list)):
            v = new_list[j]
            if u < v:
                if v % u == 0:
                    return False
            else:
                if u % v == 0:
                    return False
    return True
