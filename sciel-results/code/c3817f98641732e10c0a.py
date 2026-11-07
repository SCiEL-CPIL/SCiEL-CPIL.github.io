import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    n_max = 256
    # We'll greedily build a set, trying to add the smallest available number
    # that maintains the cardinality-separation property.
    
    # State: for each cardinality k, the set of achievable sums
    # Start with empty set: sum 0 at cardinality 0
    sums_by_card = [set() for _ in range(61)]
    sums_by_card[0] = {0}
    
    chosen = []
    used = set()
    
    # Candidate numbers to try
    candidates = list(range(1, 257))
    
    for num in candidates:
        if time.time() > deadline:
            break
        if num in used:
            continue
        
        # Try adding num: check if property is maintained
        # New sums: for each k, new_sums[k] = old_sums[k] ∪ {s + num for s in old_sums[k-1]}
        # We need: for all k != l, new_sums[k] ∩ new_sums[l] = ∅
        
        # Build new sums
        new_sums = [set() for _ in range(len(sums_by_card) + 1)]
        for k in range(len(sums_by_card)):
            for s in sums_by_card[k]:
                new_sums[k].add(s)
                if k + 1 < len(new_sums):
                    new_sums[k + 1].add(s + num)
        
        # Check cardinality separation
        valid = True
        for k in range(len(new_sums)):
            for l in range(k + 1, len(new_sums)):
                if new_sums[k] & new_sums[l]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            sums_by_card = new_sums
            chosen.append(num)
            used.add(num)
    
    # Ensure all elements are Python int
    result = [int(x) for x in chosen]
    return result
