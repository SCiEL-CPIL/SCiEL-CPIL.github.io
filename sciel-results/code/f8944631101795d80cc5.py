import time

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s limit
    
    TARGET = 200
    MAX_ELE = 100
    MASK = (1 << (TARGET + 1)) - 1  # bits 0 through 200
    
    def can_add(bitset, x):
        """Check if adding element x would create a subset sum of 200."""
        new_bits = (bitset << x) & MASK
        return not (new_bits & (1 << TARGET))
    
    def add_element(bitset, x):
        """Add element x to the reachable sums bitset."""
        return bitset | ((bitset << x) & MASK)
    
    def compute_bitset(subset):
        """Compute the reachable sums bitset for a given subset."""
        bitset = 1
        for x in subset:
            bitset = add_element(bitset, x)
        return bitset
    
    def try_local_search(cur, bitset, length, deadline):
        """Try to find a larger valid subset by local search with efficient bitset updates."""
        cur_set = set(cur)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements not in cur (from largest to smallest)
            for x in range(MAX_ELE, 0, -1):
                if x in cur_set:
                    continue
                if can_add(bitset, x):
                    cur.append(x)
                    cur_set.add(x)
                    length += 1
                    bitset = add_element(bitset, x)
                    improved = True
                    break
            
            if time.time() > deadline - 3.0:
                break
            
            # Try 1-removal + re-add swaps
            for i in range(len(cur)):
                if time.time() > deadline - 3.0:
                    break
                removed = cur[i]
                # Compute bitset without element at position i
                # Use prefix/suffix approach for efficiency
                # Recompute from scratch (n is at most ~50, so this is fast)
                bitset_no = 1
                for j, x in enumerate(cur):
                    if j == i:
                        continue
                    bitset_no = add_element(bitset_no, x)
                
                # Try to find a better element to swap in
                best_add = None
                best_val = removed
                for x in range(MAX_ELE, 0, -1):
                    if x in cur_set:
                        continue
                    if can_add(bitset_no, x):
                        if x > best_val:
                            best_add = x
                            best_val = x
                
                if best_add is not None and best_add != removed:
                    cur[i] = best_add
                    cur_set.discard(removed)
                    cur_set.add(best_add)
                    bitset = add_element(bitset_no, best_add)
                    improved = True
        
        return cur, bitset, length
    
    # Greedy seed: process from largest to smallest
    selected = []
    bitset = 1
    for x in range(MAX_ELE, 0, -1):
        if can_add(bitset, x):
            selected.append(x)
            bitset = add_element(bitset, x)
    
    best = list(selected)
    best_len = len(best)
    best_bitset = bitset
    
    if time.time() > deadline - 5.0:
        return best if best else [MAX_ELE]
    
    # Try to improve by local search
    result, result_bitset, result_len = try_local_search(best, best_bitset, best_len, deadline)
    if result_len > best_len:
        best = result
        best_len = result_len
        best_bitset = result_bitset
    
    if time.time() > deadline - 5.0:
        return best if best else [MAX_ELE]
    
    # Try different initial orderings
    for start_val in [50, 75, 25]:
        if time.time() > deadline - 10.0:
            break
        
        selected2 = []
        bitset2 = 1
        for x in range(MAX_ELE, start_val, -1):
            if can_add(bitset2, x):
                selected2.append(x)
                bitset2 = add_element(bitset2, x)
        
        for x in range(start_val, 0, -1):
            if can_add(bitset2, x):
                selected2.append(x)
                bitset2 = add_element(bitset2, x)
        
        if len(selected2) > best_len:
            best = selected2
            best_len = len(selected2)
            best_bitset = bitset2
        
        result2, result2_bitset, result2_len = try_local_search(selected2, bitset2, len(selected2), deadline)
        if result2_len > best_len:
            best = result2
            best_len = result2_len
            best_bitset = result2_bitset
    
    if time.time() > deadline - 5.0:
        return best if best else [MAX_ELE]
    
    # Try randomized local search with multiple restarts
    import random
    rng = random.Random(42)
    
    for trial in range(30):
        if time.time() > deadline - 8.0:
            break
        
        candidates = list(range(1, MAX_ELE + 1))
        rng.shuffle(candidates)
        cur = []
        cur_set = set()
        bitset3 = 1
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if can_add(bitset3, x):
                cur.append(x)
                cur_set.add(x)
                bitset3 = add_element(bitset3, x)
        
        result3, result3_bitset, result3_len = try_local_search(cur, bitset3, len(cur), deadline)
        if result3_len > best_len:
            best = result3
            best_len = result3_len
            best_bitset = result3_bitset
    
    return best if best else [MAX_ELE]
