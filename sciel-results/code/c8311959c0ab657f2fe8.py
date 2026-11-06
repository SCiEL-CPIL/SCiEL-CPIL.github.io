import random
import time

def construct():
    deadline = time.time() + 110  # Leave 10s buffer for 120s limit
    best_set = []
    best_size = 0
    
    # Initial deterministic greedy to get a baseline
    chosen, counts = [], {}
    for x in range(1, 1001):
        # Check if x can be added
        valid = True
        # Sums involving x: x+y for y in chosen where y <= x, and 2*x
        # Since chosen is sorted (we add in increasing order in this specific pass), 
        # but in random shuffle, chosen is not necessarily sorted.
        # Let's just iterate all y in chosen.
        # Optimization: Only need to check y such that x+y <= 2000 (always true for x,y <= 1000)
        # The constraint is on the count of representations.
        
        # For speed, we can pre-compute the new sums and check counts.
        new_sums = [x + y for y in chosen]
        new_sums.append(2 * x)
        
        for s in new_sums:
            c = counts.get(s, 0)
            if c >= 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
            
    if len(chosen) > best_size:
        best_size = len(chosen)
        best_set = list(chosen)

    # Randomized Greedy with Restarts
    candidates_base = list(range(1, 1001))
    
    while time.time() < deadline:
        candidates = candidates_base[:]
        random.shuffle(candidates)
        
        chosen, counts = [], {}
        
        for x in candidates:
            # Check if x can be added to current chosen set
            valid = True
            # We need to check sums x+y for all y in chosen.
            # Note: The sum a+b is the same regardless of order, so we just count occurrences.
            # The condition is: for every sum s, the number of pairs (a,b) in chosen U {x} 
            # such that a+b=s and a<=b must be <= 2.
            # Adding x creates new pairs (x, y) for y in chosen, and (x, x).
            # Note: if y > x, the pair is (y, x) which is already represented as x+y.
            # So we just need to check the sums x+y for all y in chosen, and 2x.
            
            # Optimization: If len(chosen) is large, this loop is slow.
            # But 1000 elements max, so 1000*1000/2 ~ 500k ops per restart? 
            # 500k ops is fast in Python if simple.
            
            new_sums = []
            for y in chosen:
                s = x + y
                c = counts.get(s, 0)
                if c >= 2:
                    valid = False
                    break
                new_sums.append(s)
            
            if valid:
                s = 2 * x
                c = counts.get(s, 0)
                if c >= 2:
                    valid = False
                    # No need to append 2x if invalid, but we already appended others?
                    # We should only update counts if fully valid.
                else:
                    new_sums.append(s)
                    valid = True
            
            if valid:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = list(chosen)
            # Optional: If we find a very large set, we could try to polish it, 
            # but for now just record.

    return best_set
