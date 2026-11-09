import time

def construct():
    deadline = time.time() + 85.0
    
    def compute_reachable(s):
        r = {0}
        for x in s:
            new_sums = {v + x for v in r if v + x <= 200}
            r |= new_sums
        return r

    def is_valid(s):
        r = {0}
        for x in s:
            new_sums = {v + x for v in r if v + x <= 200}
            if 200 in new_sums:
                return False
            r |= new_sums
        return True

    # Start with a known valid set: {51, ..., 100}
    # Max sum of 2 elements is 199. Min sum of 4 elements is 210.
    # So no subset sums to 200.
    selected = set(range(51, 101))
    reachable = compute_reachable(selected)
    
    # Try to add elements from 1 to 50
    # We try to add them if they don't create a sum of 200.
    # Order: Try higher numbers first? Or lower?
    # Higher numbers are "safer" in terms of not combining with many others?
    # Actually, let's just iterate 1..50.
    
    candidates = list(range(50, 0, -1))
    
    for x in candidates:
        if time.time() > deadline - 10.0:
            break
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.add(x)
            reachable |= new
            
    # Local Search: Try to swap elements to increase cardinality?
    # Since we started with 50, and added some, we have > 50.
    # Can we do better?
    # The theoretical max is likely around 50-60.
    
    # Simple improvement pass: try to add any remaining
    remaining = [x for x in range(1, 51) if x not in selected]
    for x in remaining:
        if time.time() > deadline - 10.0:
            break
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.add(x)
            reachable |= new

    return list(selected)
