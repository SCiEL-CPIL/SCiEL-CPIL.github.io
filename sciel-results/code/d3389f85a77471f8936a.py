import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Try a few different deterministic and randomized strategies
    # Strategy 1: Greedy from 1 upwards (fast version of seed)
    # Strategy 2: Greedy with larger steps
    # Strategy 3: Randomized restarts with different seeds and step sizes
    
    def try_greedy(start, step, max_val=4096):
        selected = []
        # Use a set for O(1) lookups of sums? No, sums can be large but count is manageable.
        # Actually, checking uniqueness requires storing all sums.
        # For k elements, there are C(k+2, 3) = (k+2)(k+1)k/6 sums.
        # For k=80, that's 82*81*80/6 = 88560 sums. This is small enough to fit in memory.
        # The bottleneck is generating the sums for the new candidate.
        
        sums = set()
        
        # Pre-allocate or just use a list/set
        # Incremental update: when adding x, new sums are:
        # 3x
        # 2x + y for all y in selected
        # x + y + z for all y, z in selected (y <= z)
        
        # Let's use a sorted list for selected to potentially prune? 
        # No, just check collisions.
        
        current_sums = set()
        
        for x in range(start, max_val + 1, step):
            if time.time() > deadline:
                break
                
            # Generate new sums involving x
            new_sums = []
            
            # 3x
            new_sums.append(3*x)
            
            # 2x + y
            for y in selected:
                new_sums.append(2*x + y)
                
            # x + y + z
            for i in range(len(selected)):
                y = selected[i]
                for j in range(i, len(selected)):
                    z = selected[j]
                    new_sums.append(x + y + z)
            
            # Check for collisions with existing sums or within new_sums
            # If any new sum is in current_sums, collision.
            # If any sum appears twice in new_sums, collision.
            
            # Optimization: if len(new_sums) != len(set(new_sums)), collision within new.
            # But we also need to check against current_sums.
            
            # Since we just need to know if ANY collision happens, we can check:
            # 1. Are all new_sums unique?
            # 2. Do any new_sums intersect current_sums?
            
            # Combining: check if any element in new_sums is in (current_sums U new_sums_other)
            # Simpler: 
            # if len(set(new_sums)) < len(new_sums): return selected
            # if any(s in current_sums for s in new_sums): return selected
            
            # This is O(K * M) where M is number of new sums.
            # M is roughly k^2/2.
            # So each step is O(k^3).
            # Total for k=80 is sum_{i=1}^{80} i^3 ~ 80^4 / 4 = 10^6. Very fast.
            
            # Wait, the seed program was slow because it used itertools and Python loops.
            # NumPy might help, but for k<80, Python sets are quite fast.
            # The main gain is avoiding the O(N) scan for every x if we fail early?
            # No, we scan x.
            
            # Let's just implement the efficient Python version first.
            
            # Check internal collisions in new_sums
            if len(set(new_sums)) < len(new_sums):
                return selected
                
            # Check collision with existing sums
            # Convert current_sums to a set (it already is)
            # Check intersection
            if set(new_sums).intersection(current_sums):
                return selected
                
            # No collision, add x
            selected.append(x)
            current_sums.update(new_sums)
            
        return selected

    # Run several greedy attempts with different parameters
    # 1. Start 1, Step 1
    s = try_greedy(1, 1)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s.copy()
        
    if time.time() > deadline:
        return best_set

    # 2. Start 2, Step 1
    s = try_greedy(2, 1)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s.copy()
        
    if time.time() > deadline:
        return best_set

    # 3. Randomized steps
    import random
    random.seed(42)
    
    for _ in range(20): # Limit number of restarts to ensure we finish
        if time.time() > deadline:
            break
            
        # Random start and step
        start = random.randint(1, 10)
        step = random.randint(1, 5)
        
        s = try_greedy(start, step)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s.copy()
            
    return best_set
