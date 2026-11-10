import random
import time

def subset_sums(a):
    sums = [0]
    for x in a:
        sums += [s + x for s in sums]
    return sums

def is_valid(a):
    sums = subset_sums(a)
    return len(sums) == len(set(sums))

def construct():
    # Start with powers of 2
    a = [1 << i for i in range(12)]
    best_a = a[:]
    best_max = max(a)
    
    # Sort to keep track
    a.sort()
    
    start_time = time.time()
    no_improve = 0
    
    while time.time() - start_time < 50:
        # Try to reduce the largest element
        # Find index of max
        max_idx = a.index(max(a))
        current_max = a[max_idx]
        
        improved = False
        # Try smaller values for the max element
        # Random order or descending?
        # Descending is better for finding the smallest possible quickly if we are close.
        # But random might escape local minima.
        
        # Let's try a random value smaller than current_max
        # Or just decrement?
        
        # Strategy: Randomly pick a value v in [1, current_max - 1]
        # If is_valid(a with v), accept.
        
        # To be faster, we can pre-compute sums?
        # No, 4096 sums is fast enough.
        
        # Try 100 random moves per iteration
        for _ in range(100):
            v = random.randint(1, current_max - 1)
            if v == current_max: continue
            old_a = a[:]
            a[max_idx] = v
            if is_valid(a):
                if max(a) < best_max:
                    best_max = max(a)
                    best_a = a[:]
                    a.sort()
                    improved = True
                    break
            a = old_a
            
        if not improved:
            # Stuck. Try to modify other elements to allow reduction of max?
            # Or just accept that we might be at a local min for this max.
            # Try to reduce a different element?
            # Or random restart with a smaller max bound?
            
            # If we are stuck, maybe try to swap two elements?
            # Or just break if we've been stuck for a long time?
            
            # Let's try to reduce the SECOND largest element to free up space?
            # This is complex.
            
            # Simple fallback: Random restart with a new random set in [1, best_max]
            if no_improve > 100:
                # Generate random set
                a = random.sample(range(1, best_max + 1), 12)
                if is_valid(a):
                    if max(a) < best_max:
                        best_max = max(a)
                        best_a = a[:]
                no_improve = 0
            else:
                no_improve += 1

    return best_a
