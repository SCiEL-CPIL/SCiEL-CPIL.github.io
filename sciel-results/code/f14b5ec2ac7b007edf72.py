import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Run multiple randomized greedy attempts
    while time.time() < deadline:
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        current_set = []
        counts = {}
        
        for x in candidates:
            # Check if adding x is valid
            valid = True
            
            # Check sums x + y for all y in current_set
            # Note: counts holds the number of representations for each sum
            # using pairs (a, b) with a <= b from current_set.
            # When we add x, we create new pairs (y, x) for y in current_set (if y <= x) 
            # and (x, y) for y in current_set (if y > x). 
            # Actually, the constraint is on the sum S. 
            # The number of pairs {a,b} from the set such that a+b=S is what matters.
            # If we add x, the new pairs are {x, y} for all y in current_set, and {x, x}.
            # So we check if counts[x+y] + 1 > 2 for any y in current_set, 
            # and if counts[2*x] + 1 > 2.
            
            for y in current_set:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            
            if valid:
                s2 = 2 * x
                if counts.get(s2, 0) >= 2:
                    valid = False
            
            if valid:
                # Update counts before appending to keep logic clean, or append and then update.
                # Let's update counts for the new sums involving x.
                # The new sums are x+y for y in current_set, and 2*x.
                
                # Increment counts for x + y
                for y in current_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                
                # Increment count for 2*x
                s2 = 2 * x
                counts[s2] = counts.get(s2, 0) + 1
                
                # Now add x to the set
                current_set.append(x)
                
                # Optional optimization: if we have enough time, we could try to remove 
                # some elements to make room for better ones, but simple greedy is a start.
                # For now, just greedy construction.
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return best_set
