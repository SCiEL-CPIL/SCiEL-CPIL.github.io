import random
import time

def construct():
    start_time = time.time()
    limit = 120.0
    
    # Helper to check if a candidate x can be added to set A given sum_counts
    # A is a list/set of integers. sum_counts is a dict or list.
    # We use a list for sum_counts for speed, size 2001.
    
    def can_add(x, A, sum_counts):
        # Check 2x
        if sum_counts[2*x] >= 2:
            return False
        # Check x+y
        for y in A:
            if sum_counts[x+y] >= 2:
                return False
        return True

    def add(x, A, sum_counts):
        sum_counts[2*x] += 1
        for y in A:
            sum_counts[x+y] += 1
        A.add(x)

    def remove(x, A, sum_counts):
        sum_counts[2*x] -= 1
        for y in A:
            if y != x:
                sum_counts[x+y] -= 1
        A.remove(x)

    # Initial Greedy Construction
    def greedy_init():
        A = set()
        sum_counts = [0] * 2001
        # Use random order for diversity
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, A, sum_counts):
                add(x, A, sum_counts)
        return A, sum_counts

    best_A, best_counts = greedy_init()
    best_size = len(best_A)
    
    # Local Search / Simulated Annealing
    # We will try to add elements. If stuck, we perturb.
    
    while time.time() - start_time < limit * 0.9:
        A, sum_counts = best_A, best_counts
        # We need to work on copies to allow backtracking if we want, 
        # or we can just mutate and revert.
        # Given the cost of copying sets and arrays, let's just mutate and keep the best.
        # But "best" logic is tricky if we remove.
        
        # Let's do a simple hill climb with random restarts on the "add" phase,
        # but with a perturbation step.
        
        # Perturb: Remove some elements randomly to break local optima
        # This is risky, might drop size significantly.
        # Alternative: Just try to add random elements repeatedly.
        
        # Let's try a "Random Insertion" loop.
        # If we can't insert for a while, we restart from best_A.
        
        current_A, current_counts = set(best_A), best_counts[:] # Copy counts
        
        stuck_count = 0
        max_stuck = 500 # If we try 500 random inserts and fail, restart/perturb
        
        while time.time() - start_time < limit * 0.95:
            if stuck_count > max_stuck:
                # Perturb: Remove 10% of elements randomly
                if len(current_A) > 0:
                    to_remove = random.sample(list(current_A), max(1, len(current_A)//10))
                    for x in to_remove:
                        remove(x, current_A, current_counts)
                stuck_count = 0
                continue
            
            # Try to add a random element
            # Pick a random number from 1..1000 not in current_A
            # Efficient way: pick random 1..1000, if in A, retry (or use a list of candidates)
            # Since |A| is ~500, 50% chance of collision.
            
            x = random.randint(1, 1000)
            if x in current_A:
                continue
            
            if can_add(x, current_A, current_counts):
                add(x, current_A, current_counts)
                stuck_count = 0
                if len(current_A) > best_size:
                    best_size = len(current_A)
                    best_A = current_A
                    best_counts = current_counts
            else:
                stuck_count += 1
                
    return list(best_A)
