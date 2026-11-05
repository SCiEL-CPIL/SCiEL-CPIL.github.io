import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        if not subset:
            return True
        s_set = set(subset)
        n = len(subset)
        for i in range(n):
            a = subset[i]
            for j in range(i, n):
                c = a + subset[j]
                if c in s_set:
                    return False
        return True

    def greedy_add(S, order, C_init=None):
        """Greedy: start from C_init (or empty), add elements from S in 'order' if valid."""
        C = list(C_init) if C_init is not None else []
        C_set = set(C)
        for x in order:
            if x in C_set:
                continue
            valid = True
            for a in C:
                if (x - a) in C_set:
                    valid = False
                    break
                if (a + x) in C_set:
                    valid = False
                    break
            if valid:
                if (2 * x) in C_set:
                    valid = False
            if valid:
                C.append(x)
                C_set.add(x)
        return C

    def local_improve_add(S, C_init, max_rounds=50):
        """Try to add elements from S\C to C in multiple orders, keep best."""
        C = list(C_init)
        C_set = set(C)
        excluded = [x for x in S if x not in C_set]
        best_C = C
        best_len = len(C)
        
        for rnd in range(max_rounds):
            if time.time() > DEADLINE:
                break
            improved = False
            orders = [
                sorted(excluded, key=lambda x: abs(x), reverse=True),
                sorted(excluded, key=lambda x: x, reverse=True),
                sorted(excluded, key=lambda x: x),
                sorted(excluded, key=lambda x: abs(x)),
            ]
            rng_local = random.Random(rnd * 1000 + hash(frozenset(C)) % 10000)
            shuffled = list(excluded)
            rng_local.shuffle(shuffled)
            orders.append(shuffled)
            
            for order in orders:
                if time.time() > DEADLINE:
                    break
                new_C = greedy_add(S, order, C_init=C)
                if len(new_C) > best_len:
                    best_C = new_C
                    best_len = len(new_C)
                    improved = True
                    C = new_C
                    C_set = set(C)
                    excluded = [x for x in S if x not in C_set]
            if not improved:
                break
        return best_C

    def local_improve_swap(S, C_init, max_rounds=100):
        """Swap-based local search: remove one, add two."""
        C = list(C_init)
        C_set = set(C)
        excluded = [x for x in S if x not in C_set]
        best_C = C
        best_len = len(C)
        
        rng = random.Random(12345)
        
        for rnd in range(max_rounds):
            if time.time() > DEADLINE:
                break
            improved = False
            
            # Try swapping: remove one element from C, then try to add two from excluded
            # This can escape plateaus where pure-add gets stuck
            for _ in range(10):  # try a few swaps per round
                if time.time() > DEADLINE:
                    break
                if not C or not excluded:
                    break
                
                # Pick an element to remove
                idx = rng.randrange(len(C))
                removed = C[idx]
                C_new = C[:idx] + C[idx+1:]
                C_new_set = set(C_new)
                
                # Try to add two elements from excluded
                added_count = 0
                for x in excluded:
                    if x in C_new_set:
                        continue
                    # Check if x can be added to C_new
                    valid = True
                    for a in C_new:
                        if (x - a) in C_new_set:
                            valid = False
                            break
                        if (a + x) in C_new_set:
                            valid = False
                            break
                    if valid and (2 * x) not in C_new_set:
                        C_new.append(x)
                        C_new_set.add(x)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                # Also try adding one element if we only removed one
                if added_count < 2:
                    for x in excluded:
                        if x in C_new_set:
                            continue
                        valid = True
                        for a in C_new:
                            if (x - a) in C_new_set:
                                valid = False
                                break
                            if (a + x) in C_new_set:
                                valid = False
                                break
                        if valid and (2 * x) not in C_new_set:
                            C_new.append(x)
                            C_new_set.add(x)
                            added_count += 1
                            break
                
                # Check if the swap improved the size
                if len(C_new) > len(C):
                    C = C_new
                    C_set = C_new_set
                    excluded = [x for x in S if x not in C_set]
                    best_C = C
                    best_len = len(C)
                    improved = True
                    break
            
            # Also try pure add
            if not improved:
                new_C = local_improve_add(S, C, max_rounds=5)
                if len(new_C) > best_len:
                    C = new_C
                    C_set = set(C)
                    excluded = [x for x in S if x not in C_set]
                    best_C = C
                    best_len = len(C)
                    improved = True
            
            if not improved:
                break
        
        return best_C

    def solve_instance(S, time_budget):
        """Find best sum-free subset of S within time_budget seconds."""
        n = len(S)
        candidates = []
        
        # --- Structural candidates (fast) ---
        S_odd = [x for x in S if x % 2 != 0]
        candidates.append(S_odd)
        
        M = max(S)
        S_upper = [x for x in S if x > M / 2.0]
        candidates.append(S_upper)
        
        m = min(S)
        S_lower = [x for x in S if x < m / 2.0]
        candidates.append(S_lower)
        
        abs_vals = sorted(set(abs(x) for x in S))
        for K in abs_vals[:20]:
            S_K = [x for x in S if abs(x) > K]
            candidates.append(S_K)
        
        S_pos = [x for x in S if x > 0]
        candidates.append(S_pos)
        
        S_neg = [x for x in S if x < 0]
        candidates.append(S_neg)
        
        S_large_pos = [x for x in S if x > 0 and x > M / 3.0]
        candidates.append(S_large_pos)
        
        S_large_neg = [x for x in S if x < 0 and x < m / 3.0]
        candidates.append(S_large_neg)
        
        # --- Greedy from scratch with different orderings ---
        S_sorted_abs_desc = sorted(S, key=lambda x: abs(x), reverse=True)
        S_sorted_desc = sorted(S, key=lambda x: x, reverse=True)
        S_sorted_asc = sorted(S, key=lambda x: x)
        S_sorted_abs_asc = sorted(S, key=lambda x: abs(x))
        
        rng_greedy = random.Random(42)
        S_shuffled = list(S)
        rng_greedy.shuffle(S_shuffled)
        
        greedy_orders = [S_sorted_abs_desc, S_sorted_desc, S_sorted_asc, S_sorted_abs_asc, S_shuffled]
        
        for order in greedy_orders:
            if time.time() > start_time + time_budget * 0.3:
                break
            C = greedy_add(S, order)
            candidates.append(C)
        
        # --- Validate and pick best structural/greedy candidate ---
        best_C = []
        best_len = 0
        for C in candidates:
            if len(C) <= best_len:
                continue
            if is_sum_free(C):
                best_C = C
                best_len = len(C)
        
        # --- Local improvement on best candidate ---
        if time.time() < start_time + time_budget * 0.5:
            improved = local_improve_add(S, best_C, max_rounds=30)
            if len(improved) > best_len:
                best_C = improved
                best_len = len(improved)
        
        # --- Swap-based local search ---
        if time.time() < start_time + time_budget * 0.8:
            improved2 = local_improve_swap(S, best_C, max_rounds=50)
            if len(improved2) > best_len:
                best_C = improved2
                best_len = len(improved2)
        
        # --- More local improvement if time remains ---
        remaining = start_time + time_budget - time.time()
        if remaining > 2:
            improved3 = local_improve_add(S, best_C, max_rounds=30)
            if len(improved3) > best_len:
                best_C = improved3
                best_len = len(improved3)
            
            remaining = start_time + time_budget - time.time()
            if remaining > 2:
                improved4 = local_improve_swap(S, best_C, max_rounds=30)
                if len(improved4) > best_len:
                    best_C = improved4
                    best_len = len(improved4)
        
        return best_C

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time: give more to larger instances
    time_alloc = [30, 45, 60]  # total 135s, leaving 5s buffer
    
    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        C = solve_instance(S, time_alloc[i])
        
        # Final safety check
        if not is_sum_free(C):
            # Fallback to odds
            C = [x for x in S if x % 2 != 0]
        
        results.append(C)
    
    return results
