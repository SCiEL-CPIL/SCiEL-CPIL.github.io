import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    best_set = []
    best_size = 0
    
    # Initial greedy pass to get a baseline
    # This is fast and gives a valid set
    counts = {}
    current_set = []
    for x in range(1, 1001):
        # Check if x can be added
        valid = True
        new_sums = []
        # Sums involving x and existing elements
        # Since we add in increasing order, all existing y <= x.
        # So sums are y + x where y <= x.
        # Also 2*x.
        # We need to check if counts[s] + 1 <= 2 for all new sums s.
        # But we also need to check if any s already has count 2.
        
        # Optimization: Pre-check.
        # If 2*x is already count 2, invalid.
        # If any y+x is count 2, invalid.
        
        # Let's just compute the trial counts.
        trial_counts = dict(counts)
        for y in current_set:
            s = x + y
            trial_counts[s] = trial_counts.get(s, 0) + 1
            if trial_counts[s] > 2:
                valid = False
                break
        if valid:
            s2 = 2 * x
            trial_counts[s2] = trial_counts.get(s2, 0) + 1
            if trial_counts[s2] > 2:
                valid = False
        
        if valid:
            current_set.append(x)
            counts = trial_counts
            
    best_set = current_set
    best_size = len(current_set)
    
    # Now try randomized greedy to see if we can do better
    # We will run multiple iterations
    num_iterations = 0
    while time.time() < deadline:
        num_iterations += 1
        random.seed(num_iterations)
        
        # Shuffle the range
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        counts = {}
        current_set = []
        for x in candidates:
            # Check if x can be added
            valid = True
            trial_counts = dict(counts)
            
            # We need to check sums x+y for all y in current_set.
            # Note: y can be > x or < x.
            # The representation is a+b with a<=b.
            # If we add x, new pairs are (x, y) for all y in current_set.
            # If x <= y, sum is x+y.
            # If x > y, sum is y+x.
            # In both cases, the sum value is x+y.
            # So we just add 1 to count[x+y] for all y in current_set.
            # And 1 to count[2*x].
            
            for y in current_set:
                s = x + y
                trial_counts[s] = trial_counts.get(s, 0) + 1
                if trial_counts[s] > 2:
                    valid = False
                    break
            
            if valid:
                s2 = 2 * x
                trial_counts[s2] = trial_counts.get(s2, 0) + 1
                if trial_counts[s2] > 2:
                    valid = False
            
            if valid:
                current_set.append(x)
                counts = trial_counts
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # If we've run many iterations, we might break early if no improvement for a long time?
        # But we don't know if a better one is coming.
        # The deadline guard is the primary stopper.
        
    return best_set
