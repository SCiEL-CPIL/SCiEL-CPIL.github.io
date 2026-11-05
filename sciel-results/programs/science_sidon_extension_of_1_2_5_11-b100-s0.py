import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    required = [1, 2, 5, 11]
    max_val = 284
    
    # Helper to check if a list is a valid Sidon set
    def is_valid_sidon(seq):
        sums = set()
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                s = seq[i] + seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Helper to check if adding x to seq is valid
    def can_add(seq, x, sums_set):
        # sums_set contains all pairwise sums of seq
        for y in seq:
            s = x + y
            if s in sums_set:
                return False
        if 2*x in sums_set: # x + x
            return False
        return True

    # Precompute initial sums for required
    def get_sums(seq):
        sums = set()
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                sums.add(seq[i] + seq[j])
        return sums

    best_set = None
    best_size = 0
    
    # Initial greedy run to establish a baseline
    curr = required[:]
    curr_sums = get_sums(curr)
    for x in range(12, max_val + 1):
        if can_add(curr, x, curr_sums):
            curr.append(x)
            # Update sums
            for y in curr[:-1]:
                curr_sums.add(x + y)
            curr_sums.add(2*x)
            
    if len(curr) > best_size:
        best_size = len(curr)
        best_set = curr[:]

    # Randomized search
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    
    # We will try to build sets from scratch with random order
    # and also try to improve the best set found so far via local search
    
    while time.time() < deadline:
        # Strategy 1: Randomized Greedy
        curr = required[:]
        curr_sums = get_sums(curr)
        
        # Create a pool of candidates not in required
        pool = [x for x in range(1, max_val + 1) if x not in required]
        random.shuffle(pool)
        
        for x in pool:
            if can_add(curr, x, curr_sums):
                curr.append(x)
                for y in curr[:-1]:
                    curr_sums.add(x + y)
                curr_sums.add(2*x)
        
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
            
        # Strategy 2: Local Search on the current best set
        # Try to swap elements to free up space for larger elements or just shuffle
        # If we have a set of size K, try to find a set of size K+1.
        # This is hard directly. Instead, we can try to "kick" out one element 
        # and see if we can add two, or just restart the greedy with a bias.
        
        # A simpler local search: Randomly remove one element from best_set, 
        # then run greedy from there.
        if best_set and len(best_set) > len(required):
            temp_set = best_set[:]
            # Remove a random element (not required)
            removable = [x for x in temp_set if x not in required]
            if removable:
                remove_idx = random.randint(0, len(removable)-1)
                rem_val = removable[remove_idx]
                temp_set.remove(rem_val)
                
                temp_sums = get_sums(temp_set)
                
                # Run greedy from this state
                pool2 = [x for x in range(1, max_val + 1) if x not in temp_set]
                random.shuffle(pool2)
                
                for x in pool2:
                    if can_add(temp_set, x, temp_sums):
                        temp_set.append(x)
                        for y in temp_set[:-1]:
                            temp_sums.add(x + y)
                        temp_sums.add(2*x)
                
                if len(temp_set) > best_size:
                    best_size = len(temp_set)
                    best_set = temp_set[:]
                    # Update global best sums for future reference if needed, 
                    # but we recompute in next iteration anyway.

    # Return sorted list
    if best_set:
        return sorted(best_set)
    else:
        return sorted(required)
