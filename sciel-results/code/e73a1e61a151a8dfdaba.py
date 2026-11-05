import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 146  # safety margin before 150s kill

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

    def can_add(x, C, C_set):
        """Check if x can be added to C without violating sum-free property."""
        for a in C:
            if (x - a) in C_set:
                return False
            if (a + x) in C_set:
                return False
        if (2 * x) in C_set:
            return False
        return True

    def greedy_add(S, order, C_init=None):
        """Greedy: start from C_init (or empty), add elements from S in 'order' if valid."""
        C = list(C_init) if C_init is not None else []
        C_set = set(C)
        for x in order:
            if x in C_set:
                continue
            if can_add(x, C, C_set):
                C.append(x)
                C_set.add(x)
        return C

    def local_improve(S, C_init, max_rounds=30, time_limit=None):
        """Try to add elements from S\C to C in multiple orders, keep best."""
        C = list(C_init)
        C_set = set(C)
        excluded = [x for x in S if x not in C_set]
        best_C = C
        best_len = len(C)
        
        for rnd in range(max_rounds):
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            orders = [
                sorted(excluded, key=lambda x: abs(x), reverse=True),
                sorted(excluded, key=lambda x: x, reverse=True),
                sorted(excluded, key=lambda x: x),
                sorted(excluded, key=lambda x: abs(x)),
            ]
            # Conflict-aware: sort by number of conflicts (fewer conflicts first)
            def conflict_count(x):
                cnt = 0
                for a in C:
                    if (x - a) in C_set or (a + x) in C_set:
                        cnt += 1
                if (2 * x) in C_set:
                    cnt += 1
                return cnt
            orders.append(sorted(excluded, key=conflict_count))
            
            rng_local = random.Random(rnd * 1000 + len(C) * 7 + (hash(frozenset(C[:10])) % 10000))
            shuffled = list(excluded)
            rng_local.shuffle(shuffled)
            orders.append(shuffled)
            
            for order in orders:
                if time_limit is not None and time.time() > time_limit:
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

    def swap_improve(S, C_init, max_rounds=20, time_limit=None):
        """Try swapping: remove one element from C, add one or two from excluded."""
        C = list(C_init)
        C_set = set(C)
        best_C = C
        best_len = len(C)
        excluded = [x for x in S if x not in C_set]
        
        for rnd in range(max_rounds):
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            
            # Try removing each element and adding back from excluded
            for i in range(len(C)):
                if time_limit is not None and time.time() > time_limit:
                    break
                a_remove = C[i]
                C_no_i = C[:i] + C[i+1:]
                C_no_i_set = set(C_no_i)
                
                # Try adding single elements
                for x in excluded:
                    if time_limit is not None and time.time() > time_limit:
                        break
                    if x in C_no_i_set:
                        continue
                    if can_add(x, C_no_i, C_no_i_set):
                        new_C = C_no_i + [x]
                        if len(new_C) > best_len:
                            best_C = new_C
                            best_len = len(new_C)
                            C = new_C
                            C_set = set(C)
                            excluded = [x for x in S if x not in C_set]
                            improved = True
                            break
                
                if improved:
                    break
                
                # Try adding pairs: add x1 then x2
                if time_limit is not None and time.time() > time_limit:
                    break
                for x1 in excluded:
                    if time_limit is not None and time.time() > time_limit:
                        break
                    if x1 in C_no_i_set:
                        continue
                    if not can_add(x1, C_no_i, C_no_i_set):
                        continue
                    temp_C = C_no_i + [x1]
                    temp_set = C_no_i_set | {x1}
                    for x2 in excluded:
                        if time_limit is not None and time.time() > time_limit:
                            break
                        if x2 in temp_set or x2 == x1:
                            continue
                        if can_add(x2, temp_C, temp_set):
                            candidate = temp_C + [x2]
                            if len(candidate) > best_len:
                                best_C = candidate
                                best_len = len(candidate)
                                C = candidate
                                C_set = set(C)
                                excluded = [x for x in S if x not in C_set]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
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
        m = min(S)
        S_upper = [x for x in S if x > M / 2.0]
        candidates.append(S_upper)
        S_lower = [x for x in S if x < m / 2.0]
        candidates.append(S_lower)
        
        # Magnitude threshold at various levels
        abs_vals = sorted(set(abs(x) for x in S))
        for K in abs_vals[:15]:
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
        
        # Top/bottom halves by value
        S_sorted_val = sorted(S)
        half = len(S_sorted_val) // 2
        candidates.append(S_sorted_val[half:])
        candidates.append(S_sorted_val[:half])
        
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
            if time.time() > start_time + time_budget * 0.15:
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
        t_limit_local = start_time + time_budget * 0.5
        improved = local_improve(S, best_C, max_rounds=25, time_limit=t_limit_local)
        if len(improved) > best_len:
            best_C = improved
            best_len = len(best_C)
        
        # --- Swap improvement ---
        t_limit_swap = start_time + time_budget * 0.8
        swapped = swap_improve(S, best_C, max_rounds=15, time_limit=t_limit_swap)
        if len(swapped) > best_len:
            best_C = swapped
            best_len = len(best_C)
        
        # --- Final polish: another round of local improvement with tighter focus ---
        if time.time() < start_time + time_budget * 0.95:
            t_final = start_time + time_budget * 0.98
            polished = local_improve(S, best_C, max_rounds=10, time_limit=t_final)
            if len(polished) > best_len:
                best_C = polished
                best_len = len(polished)
        
        # Final safety check
        if not is_sum_free(best_C):
            best_C = S_odd
        
        return best_C

    results = []
    time_per_instance = (DEADLINE - (time.time() - start_time)) / 3.0
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        result = solve_instance(S, time_per_instance)
        results.append(result)
    
    return results
