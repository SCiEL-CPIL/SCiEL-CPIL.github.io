import sys
import time

def construct():
    # Start time
    start_time = time.time()
    timeout = 115  # Leave 5s buffer
    
    # Prescribed elements and forced endpoints for size 20
    # We assume size 20 is possible.
    # If not, we will find the max possible.
    
    # Initial set
    initial = [1, 2, 5, 11, 284]
    sums = set()
    for i, x in enumerate(initial):
        for y in initial[i:]:
            sums.add(x + y)
            
    best_set = initial
    
    # Candidate pool
    candidates = list(range(12, 284))
    
    def is_valid(x, current_set, current_sums):
        new_sums = set()
        for y in current_set:
            s = x + y
            if s in current_sums or s in new_sums:
                return False
            new_sums.add(s)
        s2 = 2 * x
        if s2 in current_sums or s2 in new_sums:
            return False
        new_sums.add(s2)
        return True

    def search(current_set, current_sums, depth):
        nonlocal best_set, start_time, timeout
        
        if time.time() - start_time > timeout:
            return False
            
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        if len(current_set) == 20:
            return True
            
        # Try adding elements
        # Heuristic: Try elements that are "far" from existing ones? Or just iterate?
        # Iterating 12..283 is 272 items.
        for x in candidates:
            if x in current_set:
                continue
            if is_valid(x, current_set, current_sums):
                new_set = current_set + [x]
                new_sums = current_sums | {x + y for y in current_set} | {2 * x}
                if search(new_set, new_sums, depth + 1):
                    return True
        return False

    # Run search
    search(initial, sums, 0)
    
    return best_set
