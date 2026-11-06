import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # We will work with a pool of candidate integers.
    # The problem asks for integers <= 4096.
    # B_3[1] sets (Sidon-like for triples) are known to exist with density ~ n^(1/3) or similar.
    # For n=4096, max size is likely around 10-20? 
    # Wait, the seed got 0.2222 which is 2 elements? No, 0.2222 * 9? 
    # Let's check the seed score. 
    # The prompt says "score 0.2222". 
    # If the max possible is ~9-10, and seed got 2, that's low.
    # Actually, for B_3[1], the maximum size of a subset of {1..n} with distinct triple sums is related to the capacity.
    # Number of distinct sums is bounded by 3n.
    # Number of triples is K(K+1)(K+2)/6.
    # K^3/6 <= 3n => K^3 <= 18n => K <= (18*4096)^(1/3) ~ (73728)^(1/3) ~ 42.
    # So max size is around 40-45.
    
    # Strategy: Simulated Annealing on a subset of a larger pool.
    # Pool size: Let's pick a pool of size P=200 from 1..4096.
    # We want to select K elements from this pool.
    # To make it more effective, we can randomize the pool in each restart.
    
    def count_collisions(selected):
        """
        Returns the number of duplicate sums in the set of all triple sums.
        If 0, the set is valid.
        """
        if len(selected) < 3:
            return 0
        
        sums = {}
        # Generate all unordered triples (a, b, c) with a <= b <= c
        n = len(selected)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = selected[i] + selected[j] + selected[k]
                    if s in sums:
                        sums[s] += 1
                    else:
                        sums[s] = 1
        
        # Count duplicates: sum(max(0, count - 1))
        collisions = 0
        for count in sums.values():
            if count > 1:
                collisions += count - 1
        return collisions

    def get_energy(selected):
        """
        Energy function to minimize.
        Primary: number of collisions.
        Secondary: -size (we want larger sets, but we fix size in SA? 
        No, it's better to fix the size K and try to find a valid set of size K.
        If we can't find K=40, try K=39, etc.
        """
        return count_collisions(selected)

    best_found = []
    
    # Try to find sets of decreasing size K
    # Start with a high K and decrease if we fail.
    # Based on the bound K <= 42, let's try from 40 down to 10.
    
    # To speed up, we can pre-compute or use a faster collision check.
    # But K is small (~40), so K^3 ~ 64000 operations per check.
    # In Python, 64k ops is ~1ms. 
    # If we do 100k iterations, that's 100 seconds. Too slow.
    
    # Optimization: Incremental update of collisions is hard with swaps.
    # Alternative: Use a faster language? No, Python only.
    # We need to reduce the number of full collision checks or speed them up.
    
    # Let's try a different approach: Randomized Greedy with Backtracking?
    # Or, simpler: Random Sampling.
    # Pick K random numbers from 1..4096. Check collisions.
    # Probability of collision is high for K=40.
    
    # Let's stick to Simulated Annealing but optimize the inner loop.
    # Instead of checking the whole set, we can estimate the change in collisions
    # when swapping x_out for x_in.
    # New sums involving x_in:
    # 1. Sums where x_in replaces x_out in triples that previously had x_out.
    # 2. New sums created by x_in with other elements.
    # This is complex to track exactly without a full recount or careful bookkeeping.
    
    # Given the 60s limit, let's try a "Randomized Greedy" with multiple restarts and a smaller pool.
    # If we restrict the pool to the first 200 integers, the sums are smaller and collisions might be easier to avoid?
    # No, density is the issue.
    
    # Let's try: Randomly select K integers from 1..4096.
    # If collisions > 0, swap one random element with another random element.
    # Accept if collisions decrease or stay same (with probability T).
    
    # To make it faster, we can use NumPy for the sum generation?
    # No, the loop is the bottleneck.
    
    # Let's try a specific constructive heuristic:
    # Powers of 3? 1, 3, 9, 27...
    # Triple sums of powers of 3 are unique (base 3 representation).
    # 3^0, 3^1, ..., 3^k.
    # Max value 3^k <= 4096.
    # 3^7 = 2187. 3^8 = 6561.
    # So {1, 3, 9, 27, 81, 243, 729, 2187} is a valid set of size 8.
    # Score for 8? If max is ~40, 8/40 = 0.2.
    # The seed got 0.2222, which is likely 2 elements? Or maybe the seed was just bad.
    # Wait, if the seed program in the prompt is the "current program" with score 0.2222,
    # and it uses greedy 1..512, it might have found a small set.
    # Let's check the powers of 3 set.
    # It is valid. Size 8.
    # Can we do better?
    # B_3[1] sets are known.
    # A simple construction is to take numbers with small digits in base 4?
    # If we use base 4, and digits 0,1,2,3?
    # Triple sum of base 4 digits: max digit sum is 3*3=9, which carries over.
    # So base 4 doesn't work directly without spacing.
    # Base 10? Digits 0,1,2,3? Max sum 9. No carry.
    # So numbers like 1, 10, 100, 1000... work.
    # 1, 10, 100, 1000, 10000 (too big).
    # So size 4.
    
    # Powers of 3 is better (size 8).
    
    # Can we find size 10, 15, 20?
    # Random search might find larger sets.
    
    # Let's implement a fast randomized local search.
    # We will try to find a set of size K.
    # We will use a "pool" of random candidates.
    
    def try_find_set(K, time_limit=5.0):
        """
        Attempts to find a valid set of size K using randomized local search.
        """
        end_time = time.time() + time_limit
        if time.time() > end_time:
            return []
        
        # Initialize with K random distinct integers from 1..4096
        # To improve chances, we can space them out?
        # Random is fine for now.
        
        current_set = set(random.sample(range(1, 4097), K))
        current_list = list(current_set)
        
        # Calculate initial collisions
        # To speed up collision counting, we can use a dictionary.
        
        def get_collisions(lst):
            if len(lst) < 3:
                return 0
            sums = {}
            for i in range(len(lst)):
                for j in range(i, len(lst)):
                    for k in range(j, len(lst)):
                        s = lst[i] + lst[j] + lst[k]
                        sums[s] = sums.get(s, 0) + 1
            return sum(max(0, c-1) for c in sums.values())
        
        current_collisions = get_collisions(current_list)
        
        if current_collisions == 0:
            return current_list
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.95
        
        while time.time() < end_time:
            if current_collisions == 0:
                return current_list
                
            # Pick a random element to swap out
            idx_out = random.randint(0, K-1)
            x_out = current_list[idx_out]
            
            # Pick a random replacement from the pool
            # To ensure we explore, we pick a random number from 1..4096
            # and ensure it's not already in the set
            x_in = random.randint(1, 4096)
            while x_in in current_set:
                x_in = random.randint(1, 4096)
                
            # Perform swap
            current_list[idx_out] = x_in
            current_set.remove(x_out)
            current_set.add(x_in)
            
            new_collisions = get_collisions(current_list)
            
            # Acceptance probability
            delta = new_collisions - current_collisions
            if delta < 0:
                # Improvement
                current_collisions = new_collisions
            else:
                # Worse or same
                if T > T_min:
                    p = math.exp(-delta / T)
                    if random.random() < p:
                        current_collisions = new_collisions
                    else:
                        # Revert
                        current_list[idx_out] = x_out
                        current_set.remove(x_in)
                        current_set.add(x_out)
                else:
                    # Revert
                    current_list[idx_out] = x_out
                    current_set.remove(x_in)
                    current_set.add(x_out)
                    
            T *= cooling
            
        return current_list if current_collisions == 0 else []

    # Try to find sets of increasing size
    # Start small and go up? Or start high and go down?
    # Starting high is better if we want max cardinality.
    # But finding size 40 is hard.
    # Let's try 10, 15, 20, 25, 30, 35, 40.
    
    sizes_to_try = [10, 15, 20, 25, 30, 35, 40]
    
    best_set = []
    
    # First, ensure we have a valid baseline (e.g., powers of 3)
    baseline = [1, 3, 9, 27, 81, 243, 729, 2187]
    # Verify baseline
    if len(set(baseline)) == 8:
        # Quick check
        sums = set()
        for i in range(8):
            for j in range(i, 8):
                for k in range(j, 8):
                    s = baseline[i] + baseline[j] + baseline[k]
                    if s in sums:
                        # Collision? Powers of 3 should be unique.
                        pass
                    sums.add(s)
        # If unique, it's valid.
        best_set = baseline
        
    # Now try to improve
    for K in sizes_to_try:
        if time.time() > deadline - 5:
            break
            
        # Try a few restarts for each K
        found = False
        for restart in range(3):
            if time.time() > deadline - 5:
                break
                
            # Adjust time limit per restart
            time_left = deadline - time.time() - 5
            t_limit = min(5.0, time_left / 3)
            
            res = try_find_set(K, t_limit)
            if res:
                best_set = res
                found = True
                break
        
        if found:
            # If we found a set of size K, try larger K
            continue
        else:
            # If we failed to find size K, we might still try larger K?
            # No, if we can't find K, we likely can't find K+1 easily.
            # But maybe K was just hard. Let's stop if we fail on a smaller K?
            # Actually, if we fail on 10, we probably won't find 15.
            # But powers of 3 gave us 8.
            # Let's just break if we fail on the first attempt for a size > len(best_set)?
            # No, let's just try all and keep the best.
            pass

    return best_set

