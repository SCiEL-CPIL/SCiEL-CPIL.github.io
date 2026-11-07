import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    MAX_SUM = 30 * 256  # 7680
    # Use a list of bools or a set. Sets are fast in Python for membership.
    # But iterating over multiples is the cost.
    
    def is_valid_sums(S):
        if not S:
            return True
        S_list = list(S)
        max_s = max(S_list)
        # Check multiples
        for s in S_list:
            multiple = 2 * s
            while multiple <= max_s:
                if multiple in S:
                    return False
                multiple += s
        return True

    def get_subset_sums(elements):
        sums = set()
        current_sums = {0}
        for x in elements:
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + x)
            current_sums = current_sums.union(new_sums)
        current_sums.discard(0)
        return current_sums

    candidates = list(range(1, 257))
    
    best_set = []
    best_size = 0
    
    # Strategy: Greedy with multiple random orders, biased towards larger numbers
    # Larger numbers -> Larger sums -> Fewer multiples in range -> Less likely to conflict
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Create a candidate list, biased towards larger numbers
        # We can just shuffle, or sort by value descending with some noise
        # Let's try sorting by value descending, then shuffling the top half
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        # Sort by value descending?
        # Actually, random is fine if we do many restarts.
        # Let's try a mix: 50% random, 50% sorted descending
        if restart % 2 == 0:
            order = sorted(shuffled, reverse=True)
        else:
            order = shuffled
            
        current_set = []
        current_sums = set()
        
        for c in order:
            if len(current_set) >= 30:
                break
                
            # Check if adding c is valid
            # New sums: c, c+s for s in current_sums
            new_sums = set()
            new_sums.add(c)
            for s in current_sums:
                new_sums.add(s + c)
                
            # Check conflicts
            valid = True
            # Check new vs old
            for ns in new_sums:
                for os in current_sums:
                    if os == ns:
                        valid = False
                        break
                    if os < ns:
                        if ns % os == 0:
                            valid = False
                            break
                    else:
                        if os % ns == 0:
                            valid = False
                            break
                if not valid:
                    break
                    
            if valid:
                # Check new vs new
                ns_list = list(new_sums)
                max_ns = max(ns_list)
                for ns in ns_list:
                    multiple = 2 * ns
                    while multiple <= max_ns:
                        if multiple in new_sums:
                            valid = False
                            break
                        multiple += ns
                    if not valid:
                        break
                
            if valid:
                current_set.append(c)
                current_sums = current_sums.union(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]

    return best_set
