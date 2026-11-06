import time
import random

def construct():
    N = 4096
    best_set = []
    start_time = time.time()
    # Allow 85 seconds to be safe against the 90s kill
    deadline = start_time + 85.0
    
    # We will try multiple random orderings
    seed = 0
    while time.time() < deadline:
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        current_set = []
        sums_set = set()
        
        for x in candidates:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate new sums involving x
            # 1. 3x
            # 2. 2x + s for s in current_set
            # 3. x + s1 + s2 for s1, s2 in current_set, s1 <= s2
            
            new_sums = []
            new_sums.append(3 * x)
            
            for s in current_set:
                new_sums.append(2 * x + s)
            
            m = len(current_set)
            for i in range(m):
                s1 = current_set[i]
                for j in range(i, m):
                    s2 = current_set[j]
                    new_sums.append(x + s1 + s2)
            
            # Check for internal collisions in new_sums
            # Since B_3 implies B_2, pair sums s1+s2 are unique.
            # However, we must ensure x+s1+s2 doesn't collide with 2x+s or 3x.
            # Using a set is the safest and simplest way.
            new_sums_set = set(new_sums)
            if len(new_sums_set) != len(new_sums):
                continue # Collision within new sums
            
            # Check against existing sums
            if new_sums_set & sums_set:
                continue # Collision with old sums
            
            # Accept x
            current_set.append(x)
            sums_set.update(new_sums_set)
            
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            # Optional: early exit if we hit a theoretical max? 
            # Max k is not known precisely but likely < 50 for N=4096.
            # If we find a very large set, we can stop or continue to try for more.
        
        seed += 1
        
    return best_set
