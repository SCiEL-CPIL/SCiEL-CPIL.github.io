import random
import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0
    n = 4096
    best_set = []
    best_size = 0
    
    # Pre-allocate or just use standard lists/sets
    # We will run multiple randomized greedy trials
    
    while time.time() - start_time < time_limit:
        # Create a random order of candidates
        # To potentially find larger sets, we might want to bias towards smaller numbers
        # but random shuffle is a good baseline.
        # Another strategy: sort by random key, but maybe keep some structure?
        # Let's try pure random shuffle first.
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = set()
        
        # To speed up, we can break early if we've already found a set larger than best_size
        # and we are past the point where we could possibly beat it?
        # No, we don't know if we can add more.
        
        for x in candidates:
            # Check time limit
            if time.time() - start_time > time_limit - 0.5:
                break
            
            # Check if x can be added
            # New sums:
            # 1. 3x
            # 2. 2x + y for y in current_set
            # 3. x + y + z for y, z in current_set, y <= z
            
            valid = True
            
            # Check 3x
            if 3 * x in current_sums:
                valid = False
            else:
                # Check 2x + y
                for y in current_set:
                    if (2 * x + y) in current_sums:
                        valid = False
                        break
                
                if valid:
                    # Check x + y + z
                    # Use indices to avoid creating tuples if possible, though tuple creation is cheap
                    # Nested loops are faster than combinations_with_replacement for small sets
                    len_s = len(current_set)
                    for i in range(len_s):
                        y = current_set[i]
                        for j in range(i, len_s):
                            z = current_set[j]
                            if (x + y + z) in current_sums:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                # Add x to current_set
                current_set.append(x)
                
                # Update current_sums
                current_sums.add(3 * x)
                for y in current_set:
                    # Note: y includes x now, so this covers 2x+x=3x again (duplicate in set, ok)
                    # and 2x+y for y < x
                    # Wait, the loop `for y in current_set` includes x itself.
                    # 2x + x = 3x. Already added.
                    # 2x + y for y != x.
                    # This is correct.
                    current_sums.add(2 * x + y)
                
                len_s = len(current_set)
                for i in range(len_s):
                    y = current_set[i]
                    for j in range(i, len_s):
                        z = current_set[j]
                        current_sums.add(x + y + z)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # If we reached a very high score, we might want to switch to a local search
        # but given the time budget, more random trials is a good strategy.
        
    return sorted(best_set)
