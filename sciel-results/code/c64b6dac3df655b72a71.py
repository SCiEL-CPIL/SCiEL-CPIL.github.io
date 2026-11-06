"""Direct construction: take {66, 67, ..., 100} plus greedy additions from [1,65] that don't create sum 200."""
import time

def construct():
    deadline = time.time() + 55.0  # safety margin before 90s kill
    
    # Base construction: {66, 67, ..., 100}
    # Proof: any subset of size >= 3 sums to >= 66+67+68 = 201 > 200.
    # Any subset of size 2 sums to at most 99+100 = 199 < 200.
    # Any single element is <= 100 < 200.
    # Therefore no subset sums to 200.
    result = list(range(66, 101))  # {66, 67, ..., 100}, 35 elements
    
    # Compute reachable sums for the base set
    reachable = {0}
    for x in result:
        new_sums = {s + x for s in reachable if s + x <= 200}
        if 200 in new_sums:
            result.remove(x)
            continue
        reachable |= new_sums
    
    # Try to add elements from [1, 65] greedily (smallest first to maximize count)
    # An element x can be added if no reachable sum s exists with s + x == 200,
    # i.e., 200 - x is not in reachable.
    for x in range(1, 66):
        if time.time() > deadline:
            break
        complement = 200 - x
        if complement not in reachable:
            # Adding x is safe: no existing subset sums to complement
            result.append(x)
            # Update reachable sums
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
    
    if not result:
        return list(range(66, 101))
    
    return result
