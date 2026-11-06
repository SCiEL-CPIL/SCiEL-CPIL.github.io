import random
import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for 60s limit
    n = 4096
    
    best_set = []
    best_size = 0
    
    # Helper to run one greedy pass
    def run_greedy(candidates):
        selected = []
        current_sums = set()
        
        for x in candidates:
            # Generate new sums involving x
            new_sums = set()
            
            # 3x
            new_sums.add(3 * x)
            
            # 2x + y
            for y in selected:
                new_sums.add(2 * x + y)
                
            # x + 2y
            for y in selected:
                new_sums.add(x + 2 * y)
                
            # x + y + z
            # We need to iterate pairs (y,z) from selected
            # To avoid duplicates in generation (though set handles it), 
            # we just generate all.
            # Optimization: if len(selected) is large, this is the bottleneck.
            for i in range(len(selected)):
                y = selected[i]
                # y <= z
                for j in range(i, len(selected)):
                    z = selected[j]
                    new_sums.add(x + y + z)
            
            # Check for conflicts
            if new_sums & current_sums:
                continue
            
            # No conflict, add x
            selected.append(x)
            current_sums |= new_sums
            
        return selected

    # Initial deterministic run to establish a baseline
    # Sorting ascending is a common starting point
    baseline = run_greedy(list(range(1, n + 1)))
    if len(baseline) > best_size:
        best_size = len(baseline)
        best_set = baseline

    # Randomized restarts
    # We will shuffle the range and run greedy
    # To add variety, we can also try different random seeds or shuffling strategies
    
    while time.time() - start_time < time_limit:
        # Shuffle candidates
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        current_set = run_greedy(candidates)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # If we reached a very high size, we might stop or continue to try for even higher
            # Theoretical max is ~42. If we hit 42, we can stop.
            if best_size >= 42:
                break

    return best_set
