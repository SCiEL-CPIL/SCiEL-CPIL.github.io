import time

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s limit
    
    # Bitset approach: bit i represents whether sum i is reachable
    # We track reachable sums up to 200
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
    
    def is_valid(subset):
        """Check if no subset of subset sums to 200."""
        bitset = 1  # only sum 0 is reachable initially
        for x in subset:
            if not can_add(bitset, x):
                return False
            bitset = add_element(bitset, x)
        return True
    
    def try_local_search(cur, bitset, length, deadline):
        """Try to find a larger valid subset by local search."""
        cur_set = set(cur)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding elements not in cur
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
            
            if time.time() > deadline - 2.0:
                break
            
            # Try removing and re-adding to find better configurations
            for i in range(len(cur)):
                if time.time() > deadline - 2.0:
                    break
                removed = cur.pop(i)
                cur_set.discard(removed)
                length -= 1
                # Recompute reachable without removed
                bitset = 1
                for x in cur:
                    bitset = add_element(bitset, x)
                
                # Try adding larger elements
                best_add = None
                for x in range(MAX_ELE, removed, -1):
                    if x in cur_set:
                        continue
                    if can_add(bitset, x):
                        best_add = x
                        break
                
                if best_add is not None:
                    cur.append(best_add)
                    cur_set.add(best_add)
                    length += 1
                    bitset = add_element(bitset, best_add)
                    improved = True
                else:
                    # Put it back
                    cur.insert(i, removed)
                    cur_set.add(removed)
                    length += 1
                    bitset = add_element(bitset, removed)
        
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
        
        # Try a different greedy: include elements > start_val first, then fill with small
        selected2 = []
        bitset2 = 1
        # First add all large elements
        for x in range(MAX_ELE, start_val, -1):
            if can_add(bitset2, x):
                selected2.append(x)
                bitset2 = add_element(bitset2, x)
        
        # Then fill with smaller elements
        for x in range(start_val, 0, -1):
            if can_add(bitset2, x):
                selected2.append(x)
                bitset2 = add_element(bitset2, x)
        
        if len(selected2) > best_len:
            best = selected2
            best_len = len(selected2)
            best_bitset = bitset2
        
        # Local search from this seed
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
        
        # Random order greedy
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
        
        # Local search
        result3, result3_bitset, result3_len = try_local_search(cur, bitset3, len(cur), deadline)
        if result3_len > best_len:
            best = result3
            best_len = result3_len
            best_bitset = result3_bitset
    
    return best if best else [MAX_ELE]
