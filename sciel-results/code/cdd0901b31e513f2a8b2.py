import time

def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200.
    Maximize cardinality.
    """
    deadline = time.time() + 85.0
    
    def bitset_reachable(elements):
        """Compute all achievable subset sums up to 200 using bitset."""
        bits = 1  # bit 0 set: sum 0 is achievable
        for x in elements:
            if x <= 200:
                bits |= (bits << x)
        # Mask to only keep bits up to position 200
        mask = (1 << 201) - 1
        return bits & mask
    
    def has_sum_200(elements):
        """Check if any subset sums to 200."""
        bits = 1
        for x in elements:
            if x <= 200:
                bits |= (bits << x)
        return (bits >> 200) & 1 == 1
    
    def greedy(order):
        """Greedy: add elements in given order if they don't create sum 200."""
        selected = []
        bits = 1  # bitset of achievable sums
        mask = (1 << 201) - 1
        for x in order:
            # Check if adding x creates sum 200
            # This happens if (200 - x) is achievable in current set
            if (200 - x) >= 0 and ((bits >> (200 - x)) & 1):
                continue
            # Safe to add
            selected.append(x)
            bits |= (bits << x)
            bits &= mask
        return selected
    
    # Try multiple orderings
    best_set = set()
    best_size = 0
    
    # Order 1: ascending
    s1 = greedy(range(1, 101))
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = set(s1)
    
    # Order 2: descending
    s2 = greedy(range(100, 0, -1))
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = set(s2)
    
    # Order 3: large first, then small (to leave room)
    # Include all > 100? No, max is 100. 
    # Try: numbers 100 down to 1, but skip those that block many others
    s3 = greedy(list(range(100, 0, -1)))
    if len(s3) > best_size:
        best_size = len(s3)
        best_set = set(s3)
    
    # Order 4: Try to include as many as possible by being selective
    # Key insight: if I include all numbers 1..k, max sum is k(k+1)/2
    # For k=19: 19*20/2 = 190 < 200, so all 1..19 is safe
    # Can I add more? 1..19 plus some larger numbers
    # Let's try: start with 1..19, then try adding 20, 21, ... 
    base = list(range(1, 20))
    bits = 1
    mask = (1 << 201) - 1
    for x in base:
        bits |= (bits << x)
        bits &= mask
    
    extended = list(base)
    for x in range(20, 101):
        if (200 - x) >= 0 and ((bits >> (200 - x)) & 1):
            continue
        extended.append(x)
        bits |= (bits << x)
        bits &= mask
    
    if len(extended) > best_size:
        best_size = len(extended)
        best_set = set(extended)
    
    # Order 5: Try starting with larger numbers
    # Include 51..100 first (60 numbers), then try to add smaller ones
    base5 = list(range(51, 101))
    bits5 = 1
    for x in base5:
        bits5 |= (bits5 << x)
        bits5 &= mask
    
    extended5 = list(base5)
    for x in range(50, 0, -1):
        if (200 - x) >= 0 and ((bits5 >> (200 - x)) & 1):
            continue
        extended5.append(x)
        bits5 |= (bits5 << x)
        bits5 &= mask
    
    if len(extended5) > best_size:
        best_size = len(extended5)
        best_set = set(extended5)
    
    # Order 6: Include all even numbers (50 numbers), then try odds
    # Even numbers: 2,4,...,100. Sum of all evens = 2550. 
    # But we need to check if any subset sums to 200.
    # 200 is even, so it could be a sum of evens.
    base6 = list(range(2, 101, 2))
    bits6 = 1
    for x in base6:
        bits6 |= (bits6 << x)
        bits6 &= mask
    
    extended6 = list(base6)
    for x in range(1, 101, 2):
        if (200 - x) >= 0 and ((bits6 >> (200 - x)) & 1):
            continue
        extended6.append(x)
        bits6 |= (bits6 << x)
        bits6 &= mask
    
    if len(extended6) > best_size:
        best_size = len(extended6)
        best_set = set(extended6)
    
    # Local search: try to improve by swapping
    excluded = set(range(1, 101)) - best_set
    
    def compute_bits(elements):
        bits = 1
        mask = (1 << 201) - 1
        for x in elements:
            if x <= 200:
                bits |= (bits << x)
                bits &= mask
        return bits
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for to_remove in list(best_set):
            if time.time() > deadline:
                break
            candidate = set(best_set)
            candidate.discard(to_remove)
            bits = compute_bits(candidate)
            
            # Try adding excluded elements
            for to_add in sorted(excluded, reverse=True):
                if to_add in candidate:
                    continue
                if (200 - to_add) >= 0 and ((bits >> (200 - to_add)) & 1):
                    continue
                # Safe to add
                new_candidate = set(candidate)
                new_candidate.add(to_add)
                if len(new_candidate) > best_size:
                    best_set = new_candidate
                    best_size = len(best_set)
                    excluded = set(range(1, 101)) - best_set
                    improved = True
                    break
            if improved:
                break
    
    # Verify the result
    if not best_set:
        # Fallback: just return {1}
        return [1]
    
    # Final verification
    bits = compute_bits(best_set)
    if (bits >> 200) & 1:
        # Something went wrong, fall back to 1..19
        return list(range(1, 20))
    
    return list(best_set)
