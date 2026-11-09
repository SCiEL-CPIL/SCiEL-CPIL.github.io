import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Calculate LCM of 1 to 24
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
        
    # Create list of (value, x) sorted by value ascending
    # value = L / x
    items = []
    for x in range(1, 25):
        items.append((L // x, x))
    
    # Sort by value ascending
    items.sort(key=lambda p: p[0])
    
    # Beam search
    # State: (size, sums_set, selected_list)
    # We don't really need selected_list for the search, only for the final result.
    # But we need to reconstruct the set.
    # Storing the full set in each state is expensive.
    # Instead, we can store the indices of selected elements.
    # Or, since the order is fixed, we can just store a bitmask or a list of booleans.
    # 24 elements -> 24-bit mask.
    
    # Heuristic:
    # 1. Maximize size
    # 2. Minimize len(sums)
    
    beam_width = 100
    
    # Initial state
    # size=0, sums={0}, mask=0
    beam = [(0, {0}, 0)]
    
    for i, (v, x) in enumerate(items):
        new_beam = []
        
        for size, sums, mask in beam:
            # Check if v is in sums
            if v in sums:
                # Must skip
                # New state: size, sums, mask (unchanged)
                new_beam.append((size, sums, mask))
            else:
                # Option 1: Skip
                new_beam.append((size, sums, mask))
                
                # Option 2: Accept
                # New sums
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + v)
                new_mask = mask | (1 << i)
                new_beam.append((size + 1, new_sums, new_mask))
        
        # Prune new_beam
        # Sort by (size desc, len(sums) asc)
        # We need to be careful with sorting large sets.
        # len(sums) is O(1) if we cache it? No, len(set) is O(1) in Python.
        
        # To avoid keeping too many states with huge sums sets,
        # we can filter out states where len(sums) is too large?
        # No, let the heuristic handle it.
        
        # Optimization: If new_beam is very large, we might spend too much time sorting.
        # But beam_width is 100, so new_beam is at most 200.
        # Sorting 200 items is fast.
        
        new_beam.sort(key=lambda state: (-state[0], len(state[1])))
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
        # Time check
        if time.time() > deadline:
            break
            
    # Find the best state
    if not beam:
        return [1] # Fallback
        
    best_state = beam[0]
    best_size, best_sums, best_mask = best_state
    
    # Reconstruct the set
    result = []
    for i in range(24):
        if best_mask & (1 << i):
            result.append(items[i][1]) # x
            
    return result
