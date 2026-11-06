"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_set = []
    # Seed with a fast deterministic greedy pass to have a baseline
    # This ensures we return something valid even if random search fails immediately
    current = []
    used_sums = set()
    for x in range(1, 4097):
        # Check new sums involving x
        # Triple types: (x,x,x), (x,x,y), (x,y,y), (x,y,z) where y,z in current
        # Since we add x to current, we check sums involving x and elements in current
        # Let current be C. New sums are:
        # 1. 3*x
        # 2. 2*x + y for y in C
        # 3. x + 2*y for y in C
        # 4. x + y + z for y,z in C with y<=z
        
        # Optimization: Use set membership.
        # However, for speed in the initial seed, we can just do a simple loop
        # but limit it or break if too slow.
        # Actually, let's just do a few steps to get a baseline, then switch to random.
        if len(current) > 100: 
            break
            
        ok = True
        # Check 3*x
        s = 3*x
        if s in used_sums:
            ok = False
        else:
            # Check 2*x + y
            for y in current:
                s = 2*x + y
                if s in used_sums:
                    ok = False
                    break
            if ok:
                # Check x + 2*y
                for y in current:
                    s = x + 2*y
                    if s in used_sums:
                        ok = False
                        break
            if ok:
                # Check x + y + z
                for i, y in enumerate(current):
                    for z in current[i:]:
                        s = x + y + z
                        if s in used_sums:
                            ok = False
                            break
                    if not ok:
                        break
        
        if ok:
            # Add x to used_sums
            used_sums.add(3*x)
            for y in current:
                used_sums.add(2*x + y)
                used_sums.add(x + 2*y)
            for i, y in enumerate(current):
                for z in current[i:]:
                    used_sums.add(x + y + z)
            current.append(x)
            
            if time.time() > deadline - 20:
                break
                
    best_set = current
    best_len = len(best_set)
    
    if time.time() > deadline - 10:
        return best_set

    # Randomized Search
    # We will try to build sets from random permutations or random subsets of candidates.
    # Strategy: Pick a random starting point or random ordering, and greedily add.
    # To make it efficient, we maintain the set of sums.
    # We can also try "local search" or just multiple random greedy restarts.
    
    # A smarter random strategy:
    # 1. Generate a random permutation of a subset of [1, 4096].
    # 2. Greedily add elements from this permutation if valid.
    # 3. Keep the best result.
    
    # To speed up the check, we can pre-compute some properties or use NumPy.
    # But maintaining a Python set of sums is likely fastest for the check itself.
    
    # Let's optimize the check function.
    def try_add(x, current_list, sums_set):
        # Check if adding x creates any duplicate sum
        # New sums:
        # 3x
        # 2x + y
        # x + 2y
        # x + y + z
        
        if 3*x in sums_set:
            return False
        
        # Vectorized check for 2x + y and x + 2y?
        # current_list is a list of ints.
        # We can convert to numpy array for faster arithmetic?
        # np.array(current_list) * 2 + x -> check intersection with sums_set
        # Intersection with a Python set is O(N).
        
        arr = np.array(current_list)
        
        # 2x + y
        sums_2xy = 2*x + arr
        # x + 2y
        sums_x2y = x + 2*arr
        
        # Check if any of these are in sums_set
        # Converting sums_set to a numpy array for intersection is expensive if large.
        # sums_set size is ~ N^2/2. For N=50, size ~ 1250.
        # For N=100, size ~ 5000.
        # For N=200, size ~ 20000.
        
        # It might be faster to just iterate in Python for small N, 
        # but for larger N, NumPy vectorization helps.
        
        # Let's check 2x+y
        for s in sums_2xy:
            if s in sums_set:
                return False
        
        for s in sums_x2y:
            if s in sums_set:
                return False
        
        # x + y + z
        # This is the most expensive part: O(N^2)
        # We can do this with NumPy broadcasting?
        # arr[:, None] + arr[None, :] + x
        # Then check if any value is in sums_set.
        
        # For N=100, 100x100 matrix = 10k checks.
        # For N=200, 40k checks.
        
        # Let's use a hybrid: if N is small, loop. If N is large, vectorize.
        # Actually, the Python loop for N^2 is quite slow.
        # Let's vectorize the final check.
        
        # Create matrix of sums
        # To save memory/time, we can chunk or just do it if N < 100.
        # If N is large, we might skip the vectorization and just fail fast?
        # No, we need correctness.
        
        # Let's try vectorized intersection.
        # sums_3 = arr[:, None] + arr[None, :] + x
        # We only need to check unique pairs y<=z to match the definition?
        # No, the set of sums is just a set. If any sum (even from y>z) is in the set, it's a collision.
        # Wait, the sums set contains sums from UNORDERED triples.
        # So if (x, y, z) is a collision, it means x+y+z == a+b+c for some a,b,c in current.
        # The sums_set contains all such sums.
        # So checking if ANY x+y+z is in sums_set is correct.
        
        if len(current_list) > 50:
             # Vectorized
             # To avoid huge memory, we can do it in chunks or just accept the cost.
             # 200x200 int64 array is 320KB. Fine.
             sums_3 = arr[:, None] + arr[None, :] + x
             # Check intersection
             # Convert sums_set to a set for O(1) lookup, but we are iterating over sums_3...
             # If we convert sums_set to a numpy array, we can use np.isin.
             # np.isin(sums_3, sums_set_array)
             # Building sums_set_array every time is O(N^2).
             
             # Alternative: Just iterate over the flat array?
             # for s in sums_3.flat: if s in sums_set: return False
             # This is O(N^2) in Python, which is slow.
             
             # Let's stick to the Python loop for the N^2 part if N is not huge,
             # but optimize the N parts.
             # Actually, the previous seed was limited by O(N^3) recomputation.
             # This is O(N^2) check.
             
             # Let's try to speed up the N^2 check using a set of sums and vectorized generation?
             # No, the bottleneck is the lookup.
             
             # Let's just do the Python loops but break early.
             pass
             
        for i in range(len(current_list)):
            y = current_list[i]
            for j in range(i, len(current_list)):
                z = current_list[j]
                if x + y + z in sums_set:
                    return False
        
        return True

    def update_sums(x, current_list, sums_set):
        sums_set.add(3*x)
        for y in current_list:
            sums_set.add(2*x + y)
            sums_set.add(x + 2*y)
        for i in range(len(current_list)):
            y = current_list[i]
            for j in range(i, len(current_list)):
                z = current_list[j]
                sums_set.add(x + y + z)

    # Randomized Restart Loop
    while time.time() < deadline - 5:
        # Random strategy: Pick a random subset of candidates and sort them, or random permutation.
        # Let's try: Pick a random start value, then greedily pick the smallest valid number from a random pool?
        # Or: Generate a random permutation of 1..4096 and greedily add.
        
        # To increase diversity, we can bias the random permutation.
        # Let's just shuffle the range 1..4096.
        
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        current = []
        sums_set = set()
        
        for x in candidates:
            if len(current) >= best_len + 5:
                # If we are already significantly better, we can stop this restart early?
                # No, we want to maximize. But if we hit the limit, we should check.
                pass
                
            if try_add(x, current, sums_set):
                current.append(x)
                update_sums(x, current, sums_set)
                
            if len(current) > best_len:
                best_set = current
                best_len = len(best_set)
                
            # Early stop if we've gone through many candidates without improvement?
            # No, just let it run until the candidate list is exhausted or time is up.
            
            if time.time() > deadline - 5:
                break

    return best_set

# EVOLVE-BLOCK-END
