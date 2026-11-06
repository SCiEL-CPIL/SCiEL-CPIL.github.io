import random
import numpy as np

def construct():
    N_range = 129 # Sums from 128 to 256 inclusive
    S_min = 128
    S_max = 256
    U = 128 # Universe [0, 128]
    
    # Target size K
    # We want to balance K and MaxMult.
    # Smaller K generally leads to higher MaxMult, but we can tune.
    # Let's try K around 50-60.
    
    def get_multiplicities(A_set):
        counts = np.zeros(S_max - S_min + 1, dtype=np.int32)
        A_list = list(A_set)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            for j in range(i, n):
                b = A_list[j]
                s = a + b
                if S_min <= s <= S_max:
                    counts[s - S_min] += 1
        return counts

    def is_covered(counts):
        return np.all(counts > 0)

    def local_search(initial_A, steps=5000, initial_temp=10.0, cooling=0.99):
        A = set(initial_A)
        counts = get_multiplicities(A)
        if not is_covered(counts):
            return None, float('inf')
        
        current_max = np.max(counts)
        current_sq_sum = np.sum(counts**2)
        
        # Precompute indices for A
        A_list = list(A)
        
        # We will use a simple random swap strategy with annealing
        temp = initial_temp
        best_A = set(A)
        best_max = current_max
        best_sq = current_sq_sum
        
        for step in range(steps):
            if len(A) == 0: break
            a = random.choice(list(A))
            # Choose a random candidate b not in A
            # To speed up, we can precompute all candidates? No, A changes.
            # Just pick random from [0, 128] and check if in A.
            b = random.randint(0, U)
            if b in A:
                continue
            
            # Calculate delta for swap a -> b
            # Affected sums: 2a, a+x (x in A, x!=a), 2b, b+x (x in A, x!=a)
            # We only care about sums in [128, 256]
            
            # To avoid creating sets every time, we can just iterate A
            # But we need to know the change for each specific sum.
            # Let's collect the indices of affected sums in the counts array.
            affected_indices = set()
            
            s_aa = 2 * a
            if S_min <= s_aa <= S_max:
                affected_indices.add(s_aa - S_min)
            
            s_bb = 2 * b
            if S_min <= s_bb <= S_max:
                affected_indices.add(s_bb - S_min)
                
            for x in A:
                if x == a: continue
                s_ax = a + x
                if S_min <= s_ax <= S_max:
                    affected_indices.add(s_ax - S_min)
                s_bx = b + x
                if S_min <= s_bx <= S_max:
                    affected_indices.add(s_bx - S_min)
            
            if not affected_indices:
                continue

            # Calculate new counts for affected indices
            valid = True
            new_max = current_max
            new_sq_sum = current_sq_sum
            old_vals = {}
            new_vals = {}
            
            # We need to update counts in place if accepted, so we store old values
            # and calculate new values.
            
            # First pass: calculate deltas
            # Delta for a sum S:
            # -1 if S == 2a
            # -1 for each x in A\{a} such that S == a+x
            # +1 if S == 2b
            # +1 for each x in A\{a} such that S == b+x
            
            # This is O(K) per sum.
            # Number of affected sums is O(K).
            # Total O(K^2).
            
            for idx in affected_indices:
                S = idx + S_min
                old_c = counts[idx]
                old_vals[idx] = old_c
                
                change = 0
                if S == 2 * a:
                    change -= 1
                for x in A:
                    if x == a: continue
                    if S == a + x:
                        change -= 1
                    if S == b + x:
                        change += 1
                if S == 2 * b:
                    change += 1
                    
                new_c = old_c + change
                if new_c < 0:
                    valid = False
                    break
                new_vals[idx] = new_c
                
                # Update max and sq_sum tentatively
                # We can't easily update max incrementally without a histogram, 
                # but we can just compute it at the end if valid?
                # Or update new_max/new_sq_sum here?
                # If we update new_sq_sum here, we need to be careful.
                # Let's just compute new_max and new_sq_sum from the modified counts array
                # after applying the changes, only if valid.
                
            if not valid:
                continue
                
            # Apply changes to a temporary copy? No, just update in place and revert if needed.
            # But we need to know if it's better first.
            # Let's calculate new_max and new_sq_sum efficiently.
            # Since only `affected_indices` changed, we can update:
            # new_sq_sum = current_sq_sum - sum(old^2) + sum(new^2)
            delta_sq = 0
            for idx in affected_indices:
                delta_sq += new_vals[idx]**2 - old_vals[idx]**2
            new_sq_sum = current_sq_sum + delta_sq
            
            # For new_max, we need to check if any new_val is > current_max
            # and if current_max was reduced (i.e., the only element with current_max was affected and reduced).
            # This is tricky. 
            # Simple approach: If new_sq_sum < current_sq_sum, it's likely better.
            # But we specifically want to minimize MAX.
            
            # Let's just compute the new max from the affected values and the rest.
            # If no affected value is >= current_max, and the old max value was not in affected (or was reduced),
            # then new_max <= current_max.
            
            # Optimization: Maintain a histogram of counts.
            # But histogram update is also O(K).
            
            # Let's just compute the max of the new values for affected indices, 
            # and check if the global max was touched.
            
            # If current_max is not in old_vals.values(), then the global max is unaffected.
            # If current_max IS in old_vals, we need to see if it persists.
            
            # This is getting complex. Let's just do a full max calculation on the affected part 
            # and assume the rest is <= current_max.
            # If max(new_vals.values()) < current_max, then new_max is either max(new_vals) or 
            # the max of the unaffected part.
            # If the max of the unaffected part is < current_max, then new_max = max(new_vals).
            # If the max of the unaffected part == current_max, then new_max = current_max.
            
            # To be safe and simple, if new_sq_sum is significantly better, accept.
            # If new_sq_sum is worse, reject.
            
            # But we want to minimize MAX.
            # Let's try: Accept if new_max < current_max.
            # To get new_max, we can check:
            # 1. Is there any count in A' that is > current_max? 
            #    Only affected counts can change. So if max(new_vals) > current_max, new_max = max(new_vals).
            # 2. If max(new_vals) <= current_max:
            #    - If current_max was present in old_vals, it might be gone.
            #    - We need to know if any *unaffected* count is == current_max.
            #    - We can maintain a `count_of_counts` histogram.
            
            # Let's add a histogram.
            # Hist size: max possible count is K.
            
            # Let's restart the function with histogram support.
            pass

        return None, float('inf')

    # This draft is getting messy. Let's write a clean version with histogram.
    
    def local_search_v2(initial_A, steps=3000):
        A = set(initial_A)
        counts = get_multiplicities(A)
        if not is_covered(counts):
            return None, float('inf')
        
        K = len(A)
        max_possible = K
        hist = np.zeros(max_possible + 1, dtype=np.int32)
        for c in counts:
            if c <= max_possible:
                hist[c] += 1
            else:
                # Should not happen
                pass
                
        current_max = np.max(counts)
        current_sq_sum = np.sum(counts**2)
        
        best_max = current_max
        best_sq = current_sq_sum
        best_A = set(A)
        
        for step in range(steps):
            a = random.choice(list(A))
            b = random.randint(0, U)
            if b in A:
                continue
            
            # Calculate affected sums
            affected = set()
            s_aa = 2 * a
            if S_min <= s_aa <= S_max: affected.add(s_aa - S_min)
            s_bb = 2 * b
            if S_min <= s_bb <= S_max: affected.add(s_bb - S_min)
            
            for x in A:
                if x == a: continue
                s_ax = a + x
                if S_min <= s_ax <= S_max: affected.add(s_ax - S_min)
                s_bx = b + x
                if S_min <= s_bx <= S_max: affected.add(s_bx - S_min)
                
            if not affected: continue
            
            # Calculate deltas
            valid = True
            new_sq_sum = current_sq_sum
            new_hist = hist.copy() # Expensive? Size ~100. OK.
            
            # We need to update hist for each affected sum
            # hist[old_c] -= 1
            # hist[new_c] += 1
            
            # And update sq_sum
            # sq_sum += new_c^2 - old_c^2
            
            # And track if coverage is lost (new_c == 0)
            
            for idx in affected:
                old_c = counts[idx]
                S = idx + S_min
                
                change = 0
                if S == 2 * a: change -= 1
                for x in A:
                    if x == a: continue
                    if S == a + x: change -= 1
                    if S == b + x: change += 1
                if S == 2 * b: change += 1
                
                new_c = old_c + change
                if new_c < 0:
                    valid = False
                    break
                if new_c == 0 and old_c > 0:
                    valid = False
                    break
                
                new_hist[old_c] -= 1
                new_hist[new_c] += 1
                new_sq_sum += new_c*new_c - old_c*old_c
                
                # Store new_c for later application
                # We need to apply it to counts if accepted
                # Let's store in a dict
                # But we are in a loop.
                # We can just recalculate if accepted? No, we already did the work.
                # Let's store new_c in a list or dict.
                pass
            
            if not valid: continue
            
            # Find new max from new_hist
            new_max = 0
            for m in range(max_possible, -1, -1):
                if new_hist[m] > 0:
                    new_max = m
                    break
            
            # Acceptance
            accept = False
            if new_max < current_max:
                accept = True
            elif new_max == current_max and new_sq_sum < current_sq_sum:
                accept = True
            elif new_max == current_max and new_sq_sum == current_sq_sum:
                if random.random() < 0.3:
                    accept = True
            # Annealing?
            # if new_max > current_max:
            #    if random.random() < exp(-(new_max - current_max)/T):
            #        accept = True
            
            if accept:
                # Apply changes
                # We need to re-calculate new_c for each idx to apply to counts
                # This is redundant. Let's refactor to store deltas.
                
                # For now, just re-do the delta calc and apply.
                # It's O(K^2) again.
                for idx in affected:
                    old_c = counts[idx]
                    S = idx + S_min
                    change = 0
                    if S == 2 * a: change -= 1
                    for x in A:
                        if x == a: continue
                        if S == a + x: change -= 1
                        if S == b + x: change += 1
                    if S == 2 * b: change += 1
                    counts[idx] = old_c + change
                    
                hist = new_hist
                current_max = new_max
                current_sq_sum = new_sq_sum
                A.remove(a)
                A.add(b)
                
                if current_max < best_max:
                    best_max = current_max
                    best_sq = current_sq_sum
                    best_A = set(A)
                    
        return best_A, best_max

    # Main
    best_global_max = float('inf')
    best_global_A = None
    
    for trial in range(20):
        # Greedy Initialization
        A_init = set()
        # Start with some random elements to ensure spread
        # Or start with empty and add greedily?
        # Greedy add:
        # 1. Start with empty set.
        # 2. Target: cover all sums 128..256.
        # 3. Add element that covers most missing sums.
        
        # This is slow if done from scratch every time.
        # Let's use random start with size K=60.
        K = random.randint(55, 65)
        A_init = set(random.sample(range(U + 1), K))
        
        # Check coverage
        counts_init = get_multiplicities(A_init)
        if not is_covered(counts_init):
            # Try to fix by adding elements? 
            # Or just skip.
            # Let's try to add elements to cover missing sums.
            missing_sums = [s for s in range(S_min, S_max+1) if counts_init[s-S_min] == 0]
            if missing_sums:
                # Add random elements until covered or max size
                max_add = 10
                added = 0
                while missing_sums and added < max_add:
                    # Pick element that covers most missing sums
                    best_elem = -1
                    best_cover = 0
                    for e in range(U+1):
                        if e in A_init: continue
                        cover = 0
                        for ms in missing_sums:
                            # Does e help cover ms?
                            # Need x in A_init such that e+x=ms
                            # Or e+e=ms
                            # Check if (ms-e) in A_init or ms==2e
                            if (ms - e) in A_init or ms == 2*e:
                                cover += 1
                        if cover > best_cover:
                            best_cover = cover
                            best_elem = e
                    if best_elem != -1 and best_cover > 0:
                        A_init.add(best_elem)
                        added += 1
                        # Recalculate missing? Expensive.
                        # Just break and let local search handle it?
                        # Local search requires coverage.
                        # Let's just recalculate counts and missing.
                        counts_init = get_multiplicities(A_init)
                        missing_sums = [s for s in range(S_min, S_max+1) if counts_init[s-S_min] == 0]
                    else:
                        break
        
        if not is_covered(get_multiplicities(A_init)):
            continue
            
        A_res, max_res = local_search_v2(A_init, steps=2000)
        
        if A_res is not None and max_res < best_global_max:
            best_global_max = max_res
            best_global_A = A_res
            
        if best_global_max <= 12:
            break

    if best_global_A is None:
        return list(range(64, 129))
        
    return list(best_global_A)
