import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    
    # Helper to compute cost and sum counts
    # Cost is number of duplicate sums: sum(max(0, count(s) - 1))
    
    def get_initial_set(K):
        """
        Greedily construct a set of size K containing fixed elements.
        Minimizes cost at each step.
        """
        current_set = list(fixed_elements)
        current_set_set = set(current_set)
        
        # Compute initial sums and counts
        sum_counts = {}
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                s = current_set[i] + current_set[j]
                sum_counts[s] = sum_counts.get(s, 0) + 1
        
        current_cost = sum(max(0, c - 1) for c in sum_counts.values())
        
        while len(current_set) < K:
            best_candidate = None
            best_cost = float('inf')
            
            # Try a subset of candidates to save time, or all if K is small
            # For K up to 20, trying all 284 candidates is 284 * 20 ops = ~5000 ops per step.
            # 20 steps = 100k ops. Very fast.
            
            for x in range(1, 285):
                if x in current_set_set:
                    continue
                
                # Calculate new cost if we add x
                # New sums: x + y for y in current_set, and 2*x
                new_sums = [x + y for y in current_set] + [2 * x]
                
                # We need to simulate the change in cost
                # This is slightly complex because adding x changes counts of these sums.
                # Let's just compute the cost from scratch for simplicity in the greedy phase?
                # No, we can estimate.
                # Actually, recomputing cost from scratch for 284 candidates * 20 elements is:
                # 284 * (20*21/2 sums) = 284 * 210 = ~60k operations per step.
                # 20 steps = 1.2M operations. This is fine in Python? Maybe a bit slow.
                # Let's optimize by only checking candidates that don't create immediate conflicts?
                # Or just use a smaller candidate pool?
                
                # Let's try a simpler greedy: pick random candidates and keep the best?
                # No, let's just do a limited search.
                
                # To speed up, we can just check if the new sums are already in sum_counts with count >= 1.
                # If a new sum s has count 0, cost change is 0.
                # If count 1, cost change is +1.
                # If count >= 2, cost change is +1.
                # So cost change = sum(1 for s in new_sums if sum_counts.get(s, 0) >= 1)
                
                delta_cost = 0
                for s in new_sums:
                    if sum_counts.get(s, 0) >= 1:
                        delta_cost += 1
                
                if delta_cost < best_cost:
                    best_cost = delta_cost
                    best_candidate = x
                    
                    if best_cost == 0:
                        break
            
            if best_candidate is not None:
                current_set.append(best_candidate)
                current_set_set.add(best_candidate)
                
                # Update sum_counts and current_cost
                new_sums = [best_candidate + y for y in current_set[:-1]] + [2 * best_candidate]
                for s in new_sums:
                    old_count = sum_counts.get(s, 0)
                    sum_counts[s] = old_count + 1
                    if old_count >= 1:
                        current_cost += 1
            else:
                # Should not happen if K <= 20 and range is large enough
                break
                
        return current_set

    def simulate_annealing(K, initial_set, max_iter=200000, T_start=10.0, T_end=0.01):
        """
        Simulated Annealing to find a valid Sidon set of size K.
        """
        current_set = list(initial_set)
        current_set_set = set(current_set)
        
        # Compute initial sums and counts
        sum_counts = {}
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                s = current_set[i] + current_set[j]
                sum_counts[s] = sum_counts.get(s, 0) + 1
        
        current_cost = sum(max(0, c - 1) for c in sum_counts.values())
        
        if current_cost == 0:
            return current_set
        
        T = T_start
        cooling = (T_start - T_end) / max_iter
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random element to replace
            idx = random.randint(0, K - 1)
            old_elem = current_set[idx]
            
            # Pick a new element
            # To avoid picking an element already in the set, we can just pick and check
            # Or maintain a list of available elements? 
            # Picking from 1..284 and checking is fast enough.
            new_elem = random.randint(1, 284)
            
            if new_elem in current_set_set:
                continue
            
            # Calculate delta cost
            # Sums involving old_elem: old_elem + y for y in current_set \ {old_elem}, and 2*old_elem
            # Sums involving new_elem: new_elem + y for y in current_set \ {old_elem}, and 2*new_elem
            
            delta_cost = 0
            
            # Process removal of old_elem
            # For each sum s involving old_elem:
            # count[s] decreases by 1.
            # If old count was 2, new count is 1. Cost change: -1 (from 1 duplicate to 0).
            # If old count was 1, new count is 0. Cost change: 0.
            # If old count was > 2, new count is > 1. Cost change: 0 (duplicates remain).
            
            # Wait, cost is sum(max(0, count-1)).
            # If count goes n -> n-1:
            # max(0, n-1) -> max(0, n-2).
            # If n=2: 1 -> 0. Change -1.
            # If n=1: 0 -> 0. Change 0.
            # If n>2: n-1 -> n-2. Change -1.
            # So if old count >= 2, cost decreases by 1.
            
            for y in current_set:
                if y == old_elem:
                    continue
                s = old_elem + y
                old_count = sum_counts[s]
                if old_count >= 2:
                    delta_cost -= 1
            
            s_double = 2 * old_elem
            old_count = sum_counts[s_double]
            if old_count >= 2:
                delta_cost -= 1
            
            # Process addition of new_elem
            # For each sum s involving new_elem:
            # count[s] increases by 1.
            # If old count was 1, new count is 2. Cost change: +1 (0 duplicates to 1).
            # If old count was 0, new count is 1. Cost change: 0.
            # If old count was >= 2, new count is >= 3. Cost change: +1.
            # So if old count >= 1, cost increases by 1.
            
            for y in current_set:
                if y == old_elem:
                    continue
                s = new_elem + y
                old_count = sum_counts.get(s, 0)
                if old_count >= 1:
                    delta_cost += 1
            
            s_double = 2 * new_elem
            old_count = sum_counts.get(s_double, 0)
            if old_count >= 1:
                delta_cost += 1
            
            # Accept if delta_cost < 0 or with probability exp(-delta_cost/T)
            if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / max(T, 0.001)):
                # Accept move
                current_set[idx] = new_elem
                current_set_set.remove(old_elem)
                current_set_set.add(new_elem)
                
                # Update sum_counts
                for y in current_set:
                    if y == new_elem:
                        continue
                    s = old_elem + y
                    sum_counts[s] -= 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
                    
                    s = new_elem + y
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                
                s_old_double = 2 * old_elem
                sum_counts[s_old_double] -= 1
                if sum_counts[s_old_double] == 0:
                    del sum_counts[s_old_double]
                
                s_new_double = 2 * new_elem
                sum_counts[s_new_double] = sum_counts.get(s_new_double, 0) + 1
                
                current_cost += delta_cost
                
                if current_cost == 0:
                    return current_set
            
            T -= cooling
            if T < T_end:
                T = T_end
        
        return current_set

    best_set = []
    
    # Start with K=16 (known to be easy)
    # Actually, we can start with K=18 or 19 to save time, using a good initial guess.
    # Let's try to find a valid set of size 20 directly.
    
    # First, get a decent initial set of size 20 using greedy
    # Greedy might get stuck in a local minimum with cost > 0.
    # But it provides a starting point for SA.
    
    for target_K in [18, 19, 20]:
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple restarts for this K
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Generate initial set
            # Mix of greedy and random
            if restart == 0:
                init_set = get_initial_set(target_K)
            else:
                # Perturb the best found so far or generate new
                if best_set and len(best_set) == target_K - 1:
                    # Add random element to best set of size K-1
                    init_set = list(best_set)
                    while len(init_set) < target_K:
                        x = random.randint(1, 284)
                        if x not in init_set:
                            init_set.append(x)
                else:
                    init_set = get_initial_set(target_K)
                    # Perturb it
                    for _ in range(10):
                        idx = random.randint(0, len(init_set)-1)
                        if init_set[idx] not in fixed_set:
                            init_set[idx] = random.randint(1, 284)
                            while init_set[idx] in init_set[:idx] + init_set[idx+1:]:
                                init_set[idx] = random.randint(1, 284)
            
            result = simulate_annealing(target_K, init_set, max_iter=50000)
            
            # Check if result is valid
            sums = set()
            valid = True
            for i in range(len(result)):
                for j in range(i, len(result)):
                    s = result[i] + result[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            
            if valid and len(result) == target_K:
                best_set = result
                found = True
                break
        
        if found:
            # If we found size K, we can try K+1 in the next iteration
            # But we should also try to improve the current best if we didn't reach 20
            pass
        else:
            # If we didn't find a valid set of size K, keep the best partial set?
            # No, we need a valid set.
            # If we fail for K, we might fail for K+1.
            # But maybe we can find a valid set of size K-1?
            # We already have a best_set from previous iterations.
            # If best_set is empty, we should try smaller K.
            if not best_set:
                 # Fallback to smaller K
                 for smaller_K in [17, 16, 15]:
                     init_set = get_initial_set(smaller_K)
                     result = simulate_annealing(smaller_K, init_set, max_iter=50000)
                     sums = set()
                     valid = True
                     for i in range(len(result)):
                         for j in range(i, len(result)):
                             s = result[i] + result[j]
                             if s in sums:
                                 valid = False
                                 break
                             sums.add(s)
                         if not valid:
                             break
                     if valid:
                         best_set = result
                         break
            break # If we couldn't find K, don't try K+1 immediately? 
                   # Actually, if we fail K=18, we likely fail K=19.
                   # But maybe a different restart would work.
                   # Let's not break, but reduce restarts?
                   # For now, break to save time for fallback.

    if not best_set:
        # Absolute fallback
        best_set = fixed_elements
        
    return best_set
