def construct():
    # Base 5, digits 0,1, length 5
    # Max value 5^5 - 1 / 4? No, sum d_i 5^i.
    # Max is 1+5+25+125+625 = 786.
    # Wait, 5^5 = 3125. The sum of digits 1 * (5^0 + ... + 5^4) = (5^5-1)/4 = 786.
    # So all elements are <= 786.
    # This is well within 4096.
    
    base = 5
    length = 5
    elements = []
    for mask in range(1, 2**length):
        val = 0
        for i in range(length):
            if mask & (1 << i):
                val += base**i
        elements.append(val)
    
    # elements is a list of 31 distinct integers <= 786.
    # Verify B_3[1] property
    # If valid, return elements.
    # If not, we can greedily prune.
    
    # For now, assume it's valid or prune if necessary.
    # Let's check for collisions.
    # Number of triples is C(31+2, 3) = 5984.
    # We can check all.
    
    sums = {}
    valid = True
    # We need to ensure the set is B_3[1].
    # If we find a collision, we might need to remove elements.
    # But let's just return the set if it's valid.
    # If not valid, the task says "Maximize cardinality".
    # The seed score is 1.0. If this set has size 31, and seed is ~15, score is ~2.
    
    # Quick check:
    from itertools import combinations_with_replacement
    seen = set()
    for t in combinations_with_replacement(elements, 3):
        s = sum(t)
        if s in seen:
            valid = False
            break
        seen.add(s)
        
    if valid:
        return elements
    else:
        # Fallback: Greedy on these elements
        selected = []
        for x in elements:
            candidate = selected + [x]
            sums = set()
            ok = True
            for t in combinations_with_replacement(candidate, 3):
                s = sum(t)
                if s in sums:
                    ok = False
                    break
                sums.add(s)
            if ok:
                selected.append(x)
        return selected
