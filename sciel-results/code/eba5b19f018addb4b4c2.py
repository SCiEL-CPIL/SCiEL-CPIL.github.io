import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit
    
    prescribed = [1, 2, 5, 11]
    best_set = list(prescribed)
    best_size = len(best_set)
    
    # Pre-compute sums for prescribed
    def compute_sums(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums

    initial_sums = compute_sums(prescribed)
    candidates = [x for x in range(1, 285) if x not in prescribed]
    
    # Randomized Greedy with Restarts
    restarts = 0
    while time.time() < deadline:
        current_set = list(prescribed)
        current_sums = set(initial_sums)
        
        # Shuffle candidates to explore different paths
        shuffled_cands = candidates[:]
        random.shuffle(shuffled_cands)
        
        for x in shuffled_cands:
            # Check if x can be added
            # New sums: x + y for y in current_set, and 2*x
            # We need to ensure these are not in current_sums
            # Note: Since x is not in current_set, x+y are distinct for distinct y.
            # And 2*x is distinct from x+y (implies x=y, false).
            # So we just check intersection with current_sums.
            
            # Optimization: Early exit if any conflict found
            conflict = False
            # Check 2*x
            if 2 * x in current_sums:
                conflict = True
            else:
                # Check x + y
                for y in current_set:
                    if (x + y) in current_sums:
                        conflict = True
                        break
            
            if not conflict:
                current_set.append(x)
                # Update sums
                current_sums.add(2 * x)
                for y in current_set[:-1]: # y is in old set
                    current_sums.add(x + y)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            # If we hit a theoretical upper bound or a very high number, we could break early
            # Upper bound for Sidon in [1, 284] is around 17-20.
            if best_size >= 20:
                break
                
        restarts += 1
        
        # If we've done many restarts without improvement, we might try a different strategy?
        # For now, just keep going until time limit.
        
    return sorted(best_set)

# Helper for potential future use, not strictly needed inside construct if we inline
# But keeping it clean.
