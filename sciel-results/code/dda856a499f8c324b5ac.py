import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0

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

    def can_add(C, C_set, x):
        if x in C_set:
            return False
        if 2 * x in C_set:
            return False
        for c in C:
            if x + c in C_set:
                return False
            if x - c in C_set:
                return False
        return True

    def greedy_add(S, order, C_init=None):
        C = list(C_init) if C_init is not None else []
        C_set = set(C)
        for x in order:
            if x in C_set:
                continue
            if can_add(C, C_set, x):
                C.append(x)
                C_set.add(x)
        return C

    def local_improve(S, C_init, time_limit):
        """
        Improved local search:
        1. Greedy insert from multiple orders.
        2. 1-for-1 swaps to change structure (only if it enables future insert or is part of a net gain).
        3. 1-for-2 swaps (remove 1, add 2) to increase size.
        """
        C = list(C_init)
        C_set = set(C)
        excluded = [x for x in S if x not in C_set]
        excluded_set = set(excluded)
        
        best_C = C
        best_len = len(C)
        no_improve = 0
        max_no_improve = 50
        
        # Precompute orders for greedy insertion
        def get_orders(excl):
            if not excl:
                return []
            orders = [
                sorted(excl, key=lambda x: abs(x), reverse=True),
                sorted(excl, key=lambda x: x, reverse=True),
                sorted(excl, key=lambda x: x),
                sorted(excl, key=lambda x: abs(x)),
            ]
            # Add a random shuffle
            rng_local = random.Random(int(time.time() * 1000) % 10000)
            shuf = list(excl)
            rng_local.shuffle(shuf)
            orders.append(shuf)
            return orders

        while time.time() < time_limit and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Try to insert elements
            orders = get_orders(excluded)
            for order in orders:
                if time.time() > time_limit:
                    break
                new_C = greedy_add(S, order, C_init=C)
                if len(new_C) > len(C):
                    C = new_C
                    C_set = set(C)
                    excluded = [x for x in S if x not in C_set]
                    excluded_set = set(excluded)
                    improved = True
                    if len(C) > best_len:
                        best_C = list(C)
                        best_len = len(C)
                    break # Restart order generation if we improved
            
            if improved:
                no_improve = 0
                continue

            # Phase 2: Try 1-for-2 swaps (remove a, add x, add y)
            found_swap = False
            # Limit the number of removals to try per iteration to save time
            # We try a random subset of C if C is large, otherwise all
            items_to_remove = C
            if len(C) > 30:
                items_to_remove = random.sample(C, 10)
            
            for a in items_to_remove:
                if time.time() > time_limit:
                    break
                if found_swap:
                    break
                    
                # Temporarily remove a
                C_no_a = [x for x in C if x != a]
                C_no_a_set = set(C_no_a)
                
                # Try to add two elements from excluded
                # We can do this by running greedy_add on excluded with C_no_a as init
                # and checking if we get at least 2 more elements
                # But greedy_add might add more than 2.
                # We want to know if we can get size +1.
                # So we run greedy_add and see if len(new_C) > len(C_no_a) + 1? No, len(C) was len(C_no_a)+1.
                # We want len(new_C) > len(C).
                
                # To be efficient, we can just try to add 2 specific elements.
                # But greedy_add is robust. Let's use it.
                
                # Optimization: Only try this if we have enough excluded elements
                if len(excluded) < 2:
                    continue
                
                # Try a few orders for the addition
                for order in get_orders(excluded)[:2]: # Just 2 orders for speed
                    new_C = greedy_add(S, order, C_init=C_no_a)
                    if len(new_C) > len(C):
                        # Valid improvement
                        C = new_C
                        C_set = set(C)
                        excluded = [x for x in S if x not in C_set]
                        excluded_set = set(excluded)
                        improved = True
                        found_swap = True
                        if len(C) > best_len:
                            best_C = list(C)
                            best_len = len(C)
                        break
            
            if improved:
                no_improve = 0
                continue
            
            # Phase 3: Try 1-for-1 swaps (remove a, add x) to change structure
            # This is useful if it enables a future 1-for-2.
            # We accept a 1-for-1 swap only if it's "better" in some heuristic way,
            # or we accept it randomly to escape local minima, but we must ensure we don't loop.
            # Simple heuristic: Accept if the new element has larger absolute value?
            # Or just accept the first valid one found in a random order.
            
            # To prevent infinite loops, we only do this if we've been stuck for a while
            if no_improve > 5:
                items_to_remove = C
                if len(C) > 30:
                    items_to_remove = random.sample(C, 5)
                
                for a in items_to_remove:
                    if time.time() > time_limit:
                        break
                    if found_swap:
                        break
                    C_no_a = [x for x in C if x != a]
                    C_no_a_set = set(C_no_a)
                    
                    # Find any x in excluded that can be added
                    # Try random order
                    random_excluded = list(excluded)
                    random.shuffle(random_excluded)
                    
                    for x in random_excluded:
                        if can_add(C_no_a, C_no_a_set, x):
                            # Check if this swap is "good"
                            # For now, accept it. The 1-for-2 phase will try to grow from here.
                            # If it doesn't grow, we might oscillate. 
                            # To mitigate, we can keep a history of recent states or just rely on the time limit.
                            # Given the small size, oscillation is less likely to be a huge time sink if we break out.
                            C = C_no_a + [x]
                            C_set = set(C)
                            excluded = [x for x in S if x not in C_set]
                            excluded_set = set(excluded)
                            improved = True
                            found_swap = True
                            break
                            
            if not improved:
                no_improve += 1

        return best_C

    def solve_instance(S, time_budget):
        n = len(S)
        candidates = []
        
        # --- Structural candidates ---
        S_odd = [x for x in S if x % 2 != 0]
        candidates.append(S_odd)
        
        M = max(S)
        m = min(S)
        S_upper = [x for x in S if x > M / 2.0]
        candidates.append(S_upper)
        S_lower = [x for x in S if x < m / 2.0]
        candidates.append(S_lower)
        
        S_pos = [x for x in S if x > 0]
        candidates.append(S_pos)
        S_neg = [x for x in S if x < 0]
        candidates.append(S_neg)
        
        # Magnitude thresholds
        abs_vals = sorted(set(abs(x) for x in S))
        if abs_vals:
            # Try several K values
            for i in range(0, len(abs_vals), max(1, len(abs_vals)//10)):
                K = abs_vals[i]
                S_K = [x for x in S if abs(x) > K]
                candidates.append(S_K)
            # Also the largest K (smallest set) and smallest K (largest set)
            candidates.append([x for x in S if abs(x) > abs_vals[-1]])
            candidates.append([x for x in S if abs(x) > abs_vals[0]])

        # Modulo 3
        S_mod1 = [x for x in S if x % 3 == 1]
        S_mod2 = [x for x in S if x % 3 == 2]
        candidates.append(S_mod1)
        candidates.append(S_mod2)
        
        # --- Greedy from scratch ---
        S_sorted_abs_desc = sorted(S, key=lambda x: abs(x), reverse=True)
        S_sorted_desc = sorted(S, key=lambda x: x, reverse=True)
        S_sorted_asc = sorted(S, key=lambda x: x)
        S_sorted_abs_asc = sorted(S, key=lambda x: abs(x))
        
        rng_greedy = random.Random(42)
        S_shuffled = list(S)
        rng_greedy.shuffle(S_shuffled)
        
        greedy_orders = [S_sorted_abs_desc, S_sorted_desc, S_sorted_asc, S_sorted_abs_asc, S_shuffled]
        
        # Limit time for initial greedy
        t_greedy_limit = start_time + time_budget * 0.2
        for order in greedy_orders:
            if time.time() > t_greedy_limit:
                break
            C = greedy_add(S, order)
            candidates.append(C)
        
        # --- Validate and pick best candidate ---
        best_C = []
        best_len = 0
        for C in candidates:
            if len(C) <= best_len:
                continue
            if is_sum_free(C):
                best_C = C
                best_len = len(C)
        
        # --- Local improvement ---
        t_limit = start_time + time_budget
        improved = local_improve(S, best_C, t_limit)
        if len(improved) > len(best_C):
            best_C = improved
            
        return best_C

    results = []
    
    # Allocate time budget to each instance
    # 60, 120, 240 sizes. Larger sizes need more time?
    # But local search scales with n^2 roughly.
    # Let's give proportional time based on size, but with a minimum.
    # Total available: DEADLINE - start_time
    # Let's just split evenly for simplicity, or slightly favor the largest.
    # (60, 120, 240) -> weights 1, 2, 4?
    # Let's try even split first.
    
    instances = [(60, 801), (120, 802), (240, 803)]
    total_time = DEADLINE - start_time
    # Weights: 1, 2, 4
    weights = [1, 2, 4]
    total_weight = sum(weights)
    
    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        time_budget = total_time * (weights[i] / total_weight)
        # Ensure we don't exceed the global deadline
        end_time = start_time + time_budget
        # But we also need to leave time for the next ones?
        # The loop structure handles this by passing a budget.
        # However, local_improve uses time.time() < time_limit.
        # We should pass the absolute deadline for this specific instance.
        
        # Calculate absolute deadline for this instance
        # To be safe, we calculate the deadline as start_time + allocated_time
        # But if we are slow, we might overshoot.
        # Let's just use the current time + budget.
        
        # Actually, better to track remaining time.
        # Let's just pass a deadline that is start_time + (total_time * weight/total_weight)
        # This assumes perfect scheduling.
        
        # A safer approach:
        # deadline_i = start_time + (total_time * (sum(weights[:i+1]) / total_weight))
        # This ensures the last instance finishes by DEADLINE.
        
        cumulative_weight = sum(weights[:i+1])
        deadline_i = start_time + total_time * (cumulative_weight / total_weight)
        
        # But if we are already late, we should reduce the budget.
        if deadline_i > DEADLINE:
            deadline_i = DEADLINE
            
        best = solve_instance(S, deadline_i - start_time)
        results.append(best)
        
        # Check time
        if time.time() > DEADLINE - 1:
            # Emergency stop, return what we have
            # Fill remaining with empty or simple heuristic
            for j in range(i+1, len(instances)):
                size_j, seed_j = instances[j]
                S_j = instance(size_j, seed_j)
                # Just return odds
                results.append([x for x in S_j if x % 2 != 0])
            break

    return results
