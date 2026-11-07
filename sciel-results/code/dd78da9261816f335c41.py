import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Fast validity check
    def is_valid(elements):
        if not elements:
            return True
        # Generate all subset sums
        sums = [0]
        for x in elements:
            sums += [s + x for s in sums]
        sums = sums[1:] # Remove 0
        
        # Check distinctness
        if len(sums) != len(set(sums)):
            return False
        
        sum_set = set(sums)
        max_sum = max(sums)
        
        # Check divisibility
        # For each sum a, check if any multiple 2a, 3a... is in sum_set
        for a in sums:
            m = 2 * a
            while m <= max_sum:
                if m in sum_set:
                    return False
                m += a
        return True

    best_set = []
    best_size = 0
    
    # Try to find a valid set of increasing sizes
    # We can start with a greedy construction or random restarts
    
    # Greedy construction attempt
    # Try to build a set by adding elements that don't break validity
    # To speed up, we can prioritize elements that are "safe"
    
    # Let's try a randomized local search for a fixed size, then increase size
    # Start with size 10, go up to 20
    
    # Candidate pool
    candidates = list(range(1, 257))
    
    # We'll try to find a valid set of size k
    # Start with a random subset of size k
    # Then try to improve it
    
    # Actually, a simple greedy might work well if we sort candidates
    # Let's try to find a set of size 15 first
    
    for target_size in range(15, 5, -1): # Try large sizes first
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(1000): # 1000 attempts per size
            if time.time() > deadline:
                break
                
            # Random subset of target_size
            current_set = random.sample(candidates, target_size)
            
            if is_valid(current_set):
                if target_size > best_size:
                    best_set = current_set
                    best_size = target_size
                found = True
                # If we found a valid set of this size, we can stop trying this size
                # and go to the next larger size? No, we are iterating downwards.
                # So if we found size 15, we can return it or try to find 16?
                # The loop goes 15, 14, 13...
                # So if we find 15, we should break out and try 16?
                # Let's restructure: try sizes 20 down to 1.
                pass
        
        # If we found a valid set of this target_size, we can stop searching for smaller sizes?
        # No, we want the MAX size.
        # If we found a valid set of size 15, we should try to find size 16.
        # So the loop should be increasing? Or we just track the best.
        
        # Let's change strategy:
        # Try to construct a set greedily.
        pass

    # Better Strategy: Greedy with Backtracking / Randomized Greedy
    # Start with empty set.
    # Try to add elements.
    # If we get stuck, backtrack or restart.
    
    # Let's try a simple greedy that picks random elements
    # and checks if adding them keeps the set valid.
    
    best_set = []
    
    for _ in range(5000): # 5000 restarts
        if time.time() > deadline:
            break
            
        current_set = []
        available = list(range(1, 257))
        random.shuffle(available)
        
        for x in available:
            # Check if adding x to current_set is valid
            new_set = current_set + [x]
            if is_valid(new_set):
                current_set = new_set
                if len(current_set) > len(best_set):
                    best_set = current_set
        
    return best_set

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I need to wrap it in the required format.
# EVOLVE-BLOCK-END
