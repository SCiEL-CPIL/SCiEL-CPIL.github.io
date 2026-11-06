import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 90.0  # Wall clock limit
    
    # 1. Compute LCM of 1..22
    L = 1
    for i in range(1, 23):
        L = L * i // math.gcd(L, i)
        
    # 2. Prepare weights
    # We want to maximize cardinality.
    # Hypothesis: Adding smaller weights (larger denominators) first allows more elements.
    # Let's try a greedy approach with multiple ordering strategies.
    
    indices = list(range(1, 23))
    
    # Strategy 1: Smallest weights first (descending index)
    # Strategy 2: Largest weights first (ascending index)
    # Strategy 3: Randomized restarts
    
    best_set = []
    best_size = 0
    
    def check_and_update(current_sums, current_selected, w, i):
        """
        Try to add weight w (index i) to current_sums.
        Returns (success, new_sums, new_selected)
        """
        # Generate new sums
        new_sums = set()
        for s in current_sums:
            ns = s + w
            if ns in current_sums:
                return False, current_sums, current_selected
            new_sums.add(ns)
        
        # Check if new_sums has internal collisions (shouldn't if current_sums is valid)
        # Actually, if s1 != s2, then s1+w != s2+w. So internal collisions in new_sums
        # only happen if s1+w == s2+w which implies s1==s2.
        # The only collision risk is against existing sums.
        
        # Combine
        combined_sums = current_sums.union(new_sums)
        # Since we checked intersection, len should be 2 * len(current_sums)
        
        return True, combined_sums, current_selected + [i]

    # Helper to run a greedy pass given an order
    def run_greedy(order):
        sums = {0}
        selected = []
        for i in order:
            w = L // i
            success, sums, selected = check_and_update(sums, selected, w, i)
            if not success:
                continue
            # Early exit if time is running out? No, greedy is fast.
        return selected

    # Try a few deterministic orders
    orders = [
        list(range(22, 0, -1)),  # Smallest weights first
        list(range(1, 23)),      # Largest weights first
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]
    
    # Also try a "mixed" heuristic: sort by (L//i) % some_prime? No, keep it simple.
    
    for order in orders:
        candidate = run_greedy(order)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate
            
    # If we have time, do random restarts to see if we can find a larger set
    # The search space is small, but greedy might get stuck in a local optimum.
    # However, for "dissociated sets", greedy by size is often optimal or near-optimal.
    
    import random
    random.seed(42)
    
    # Time check
    if time.time() - start_time < 5.0:
        # Do a few random shuffles
        for _ in range(50):
            order = indices[:]
            random.shuffle(order)
            candidate = run_greedy(order)
            if len(candidate) > best_size:
                best_size = len(candidate)
                best_set = candidate
                
            if time.time() - start_time > 80.0:
                break

    # Return the best set found
    # Ensure it's a list of ints
    return best_set

# EVOLVE-BLOCK-END
