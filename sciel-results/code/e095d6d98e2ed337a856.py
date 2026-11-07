import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    def check_addition(current_sums, new_elem):
        """
        Returns the new set of sums if valid, else None.
        """
        # 1. Generate new sums
        new_sums = set()
        for s in current_sums:
            ns = s + new_elem
            # Collision check
            if ns in current_sums or ns in new_sums:
                return None
            new_sums.add(ns)
        
        if new_elem in current_sums or new_elem in new_sums:
            return None
        new_sums.add(new_elem)
        
        # 2. Divisibility Check
        # We only need to check interactions involving new_sums
        # because current_sums is already internally valid.
        
        # Combine for iteration
        # To avoid O(N^2) on the whole set, we iterate new_sums 
        # and check against (current_sums U new_sums).
        
        # Optimization: 
        # If s in new_sums, and t in current_sums:
        #   if t % s == 0 or s % t == 0: fail
        # If s1 in new_sums, s2 in new_sums, s1 != s2:
        #   if s2 % s1 == 0 or s1 % s2 == 0: fail
        
        # Let's just do a simple double loop on the union, 
        # but break early.
        
        # For performance, convert to list
        all_sums = list(current_sums) + list(new_sums)
        
        # We can be smarter:
        # For each s in new_sums:
        #   for each t in all_sums:
        #      if s == t: continue
        #      if t % s == 0: return None
        #      # s % t == 0 is redundant if we check all pairs? 
        #      # If we check s vs t, we cover both directions if we iterate all pairs.
        #      # But here s is fixed from new, t from all.
        #      # If t is old and s is new: checks new/old.
        #      # If t is new and s is new: checks new/new.
        #      # This covers all pairs where at least one is new.
        
        for s in new_sums:
            for t in all_sums:
                if s == t:
                    continue
                if t % s == 0:
                    return None
                # Note: if s % t == 0, it will be caught when 
                # we pick s'=t (if t is new) or if t is old?
                # If t is old, s is new. We check t % s.
                # Do we need s % t?
                # If s < t, s % t is s, not 0.
                # If s > t, s % t could be 0.
                # Example: t=2, s=4. t % s = 2. s % t = 0.
                # My loop: s=4, t=2. t % s = 2. Passes.
                # But 2 divides 4! So I MUST check s % t as well.
                
                if s % t == 0:
                    return None
        return new_sums

    best_set = []
    best_size = 0
    
    # Candidate pool: 3 to 256
    # We assume 1 and 2 are bad for size > 2.
    # If we find a set of size 2 with 1 or 2, and no larger set, we should return it.
    # But likely size > 2 exists.
    
    candidates = list(range(3, 257))
    
    # Greedy Pass 1: Sorted
    current_set = []
    current_sums = set()
    for c in candidates:
        if time.time() > deadline:
            break
        res = check_addition(current_sums, c)
        if res is not None:
            current_set.append(c)
            current_sums = current_sums.union(res)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                
    # Random Shuffles
    # We can also try including 1 and 2 in the pool for shuffles, 
    # just in case a set of size 2 is the max, or if my logic about 1,2 is slightly off for specific cases.
    # But 1 definitely blocks size > 1.
    # 2 blocks size > 2.
    # So if best_size < 3, we should try including 1,2.
    
    if best_size < 3:
        candidates = list(range(1, 257))
    
    attempts = 0
    while time.time() < deadline - 2.0:
        random.shuffle(candidates)
        current_set = []
        current_sums = set()
        for c in candidates:
            if time.time() > deadline:
                break
            res = check_addition(current_sums, c)
            if res is not None:
                current_set.append(c)
                current_sums = current_sums.union(res)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
        attempts += 1
        if attempts > 50:
            break

    return best_set
