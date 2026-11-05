import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 90s limit, leave 5s margin

    # Step 1: Build the set of all subsets of [1..100] that sum to 200
    # Using bitset DP to find reachable sums, then enumerate actual subsets
    # For efficiency, we'll compute for each element x, how many subsets of the
    # remaining elements can combine with x to make 200
    
    # Precompute: for each x in [1,100], find all subsets of [1..100]\{x} that sum to 200-x
    # This tells us which other elements are "in conflict" with x
    
    # Use bitset DP to find reachable sums
    # conflict_count[x] = number of subsets S of [1..100]\{x} such that sum(S) = 200-x
    # This is expensive to compute exactly, so use a proxy:
    # For each x, compute the set of elements that can appear in a subset summing to 200-x
    
    # Better approach: For each target t in [1, 199], find all subsets of [1..100] summing to t
    # Then for each element x, count how many such subsets contain x
    
    # This is still expensive. Let's use a different strategy:
    # Greedy with lookahead: at each step, for each candidate x, simulate adding it
    # and check if it creates a new 200-sum. Use bitset for fast reachability.
    
    # Initialize reachable sums bitset
    # Bit i is 1 if sum i is reachable
    
    # Greedy initialization with conflict-aware ordering
    # For each x, compute "danger score" = number of subsets of current set that would
    # combine with x to make 200
    
    selected = []
    reachable = 1  # bit 0 set, sum 0 is reachable
    
    # Precompute for each x in [1..100]: the set of y in [1..100] such that
    # there exists a subset of [1..100]\{x,y} summing to 200-x-y
    # This is the "conflict graph" - edge (x,y) means x and y are in some 200-sum together
    
    # Actually, let's just do greedy with dynamic conflict checking
    # At each step, for each missing element x, check if adding x creates a 200-sum
    # If not, add it. Order by some heuristic.
    
    # Heuristic: process elements in an order that minimizes conflicts
    # Start with larger elements (they have fewer combinations to reach 200)
    
    candidates = list(range(100, 0, -1))
    
    for x in candidates:
        if time.time() > deadline:
            break
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach
    
    best_set = set(selected)
    best_size = len(best_set)
    
    # Local search with bitset reachability
    current_set = best_set.copy()
    current_reachable = reachable
    
    all_elements = list(range(1, 101))
    
    def can_add(x, R):
        """Check if x can be added to set with reachable sums R without creating 200-sum"""
        return not ((R >> (200 - x)) & 1)
    
    def update_reachable(R, x):
        return R | (R << x)
    
    def is_valid_set(S):
        r = 1
        for x in S:
            r |= (r << x)
            if (r >> 200) & 1:
                return False
        return True
    
    # Phase 1: Try to add any single missing element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Sort missing elements by "safety" - elements that are less likely to create conflicts
        # Use: 200 - x (smaller values have more room to combine)
        missing = [x for x in all_elements if x not in current_set]
        missing.sort(key=lambda x: (200 - x, -x))  # Prefer larger x (safer)
        
        for x in missing:
            if time.time() > deadline:
                break
            if can_add(x, current_reachable):
                current_set.add(x)
                current_reachable = update_reachable(current_reachable, x)
                improved = True
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)
        
        # Phase 2: Directed 1-for-1 swaps
        if not improved and time.time() < deadline:
            curr_list = list(current_set)
            miss_list = [x for x in all_elements if x not in current_set]
            miss_list.sort(key=lambda x: (200 - x, -x))
            
            found_swap = False
            for a in curr_list:
                if time.time() > deadline:
                    break
                for b in miss_list:
                    if time.time() > deadline:
                        break
                    new_set = current_set.copy()
                    new_set.remove(a)
                    new_set.add(b)
                    if is_valid_set(new_set):
                        current_set = new_set
                        current_reachable = 1
                        for x in current_set:
                            current_reachable |= (current_reachable << x)
                        improved = True
                        found_swap = True
                        if len(current_set) > best_size:
                            best_set = current_set.copy()
                            best_size = len(current_set)
                        break
                if found_swap or time.time() > deadline:
                    break
            
            if found_swap:
                continue
        
        # Phase 3: 1-for-2 swaps to increase size
        if not improved and time.time() < deadline:
            curr_list = list(current_set)
            miss_list = [x for x in all_elements if x not in current_set]
            miss_list.sort(key=lambda x: (200 - x, -x))
            
            found_12 = False
            for a in curr_list:
                if time.time() > deadline:
                    break
                for i in range(len(miss_list)):
                    if time.time() > deadline:
                        break
                    b = miss_list[i]
                    for j in range(i + 1, len(miss_list)):
                        if time.time() > deadline:
                            break
                        c = miss_list[j]
                        new_set = current_set.copy()
                        new_set.remove(a)
                        new_set.add(b)
                        new_set.add(c)
                        if is_valid_set(new_set):
                            current_set = new_set
                            current_reachable = 1
                            for x in current_set:
                                current_reachable |= (current_reachable << x)
                            improved = True
                            found_12 = True
                            if len(current_set) > best_size:
                                best_set = current_set.copy()
                                best_size = len(current_set)
                            break
                    if found_12 or time.time() > deadline:
                        break
                if found_12 or time.time() > deadline:
                    break
            
            if found_12:
                continue
    
    # Final high-precision polish: exhaustive single-add pass
    if time.time() < deadline:
        for x in range(100, 0, -1):
            if x in current_set:
                continue
            if can_add(x, current_reachable):
                current_set.add(x)
                current_reachable = update_reachable(current_reachable, x)
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)
    
    # Final 1-for-2 exhaustive polish
    if time.time() < deadline:
        curr_list = sorted(current_set, reverse=True)
        miss_list = [x for x in range(1, 101) if x not in current_set]
        found = True
        while found and time.time() < deadline:
            found = False
            for a in curr_list:
                if time.time() > deadline:
                    break
                new_set_without_a = current_set - {a}
                r_without_a = 1
                for x in new_set_without_a:
                    r_without_a |= (r_without_a << x)
                for i in range(len(miss_list)):
                    if time.time() > deadline:
                        break
                    b = miss_list[i]
                    if b in current_set:
                        continue
                    if can_add(b, r_without_a):
                        r_with_b = r_without_a | (r_without_a << b)
                        for j in range(i + 1, len(miss_list)):
                            if time.time() > deadline:
                                break
                            c = miss_list[j]
                            if c in current_set:
                                continue
                            if can_add(c, r_with_b):
                                new_set = new_set_without_a | {b, c}
                                if is_valid_set(new_set):
                                    current_set = new_set
                                    current_reachable = 1
                                    for x in current_set:
                                        current_reachable |= (current_reachable << x)
                                    if len(current_set) > best_size:
                                        best_set = current_set.copy()
                                        best_size = len(current_set)
                                    found = True
                                    break
                    if found or time.time() > deadline:
                        break
                if found or time.time() > deadline:
                    break
    
    return list(best_set)
