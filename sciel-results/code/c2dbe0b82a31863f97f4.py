# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(S, S_set, x):
        if 2 * x in S_set:
            return False
        for a in S:
            if a + x in S_set:
                return False
        return True

    def greedy_augment(S, all_elements, order=None):
        S_set = set(S)
        S_list = list(S)
        if order is None:
            order = all_elements
        for x in order:
            if x not in S_set:
                if can_add(S_list, S_set, x):
                    S_set.add(x)
                    S_list.append(x)
        return S_list

    def local_search(S, all_elements, time_budget=2.0, rng=None):
        """Local search: repeatedly try to add an element, and if stuck, remove a random element and try again."""
        if rng is None:
            rng = random.Random()
        
        S = list(S)
        S_set = set(S)
        best_S = list(S)
        best_size = len(S)
        end_time = time.time() + time_budget
        
        # Build list of all elements not in S
        available = [x for x in all_elements if x not in S_set]
        
        while time.time() < end_time:
            # Try to add any available element
            added = False
            for x in available:
                if can_add(S, S_set, x):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                    added = True
                    break
            
            if not added:
                # Stuck: try removing a random element to unblock
                if len(S) > best_size // 2:  # Don't remove too much
                    idx = rng.randint(0, len(S) - 1)
                    removed = S.pop(idx)
                    S_set.remove(removed)
                    available.append(removed)
                else:
                    break
        
        return best_S

    def solve_simulated_annealing(vals, time_budget=10.0, rng=None):
        """Simulated annealing to find large sum-free subset."""
        if rng is None:
            rng = random.Random()
        
        n = len(vals)
        if n == 0:
            return []
        
        vals_set = set(vals)
        
        # Precompute sum_mask and double_mask for fast checking
        index_of_val = {v: i for i, v in enumerate(vals)}
        sum_mask = [0] * n
        double_mask = [0] * n
        
        for i in range(n):
            vi = vals[i]
            for j in range(n):
                s = vi + vals[j]
                if s in index_of_val:
                    k = index_of_val[s]
                    sum_mask[i] |= (1 << k)
            dv = 2 * vi
            if dv in index_of_val:
                k = index_of_val[dv]
                double_mask[i] |= (1 << k)
        
        def can_add_bitset(current_bits, idx):
            return (current_bits & sum_mask[idx]) == 0 and (current_bits & double_mask[idx]) == 0
        
        def count_bits(m):
            c = 0
            while m:
                c += 1
                m &= m - 1
            return c
        
        # Initialize with a good starting point: odds + greedy
        odd = [x for x in vals if x % 2 != 0]
        S = greedy_augment(odd, vals)
        S_set = set(S)
        current_bits = 0
        for x in S:
            idx = index_of_val[x]
            current_bits |= (1 << idx)
        current_size = len(S)
        
        best_bits = current_bits
        best_size = current_size
        
        T_start = time.time() + time_budget
        T0 = 1.0
        alpha = 0.999
        
        T = T0
        while time.time() < T_start:
            T *= alpha
            
            # Pick a random element to flip
            idx = rng.randint(0, n - 1)
            bit = 1 << idx
            
            if current_bits & bit:
                # Remove element
                new_bits = current_bits & ~bit
                new_size = current_size - 1
                # Always accept removals (or with high probability)
                if rng.random() < 0.8:  # 80% chance to accept removal
                    current_bits = new_bits
                    current_size = new_size
            else:
                # Add element
                if can_add_bitset(current_bits, idx):
                    new_bits = current_bits | bit
                    new_size = current_size + 1
                    current_bits = new_bits
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_bits = current_bits
                else:
                    # Can't add directly. Try accepting with probability based on T
                    # This is a heuristic: we accept "infeasible" moves to escape local optima
                    # But we must ensure we don't keep infeasible states too long
                    # Instead, let's just skip this move
                    pass
            
            # Occasionally try a multi-step move: remove one, add another
            if rng.random() < 0.3:
                idx_remove = rng.randint(0, n - 1)
                bit_remove = 1 << idx_remove
                if current_bits & bit_remove:
                    temp_bits = current_bits & ~bit_remove
                    # Try to add a random element
                    idx_add = rng.randint(0, n - 1)
                    if not (temp_bits & (1 << idx_add)):
                        if can_add_bitset(temp_bits, idx_add):
                            current_bits = temp_bits | (1 << idx_add)
                            current_size = current_size  # Size unchanged
                            if current_size > best_size:
                                best_size = current_size
                                best_bits = current_bits
        
        # Extract best solution
        result = []
        for i in range(n):
            if best_bits & (1 << i):
                result.append(vals[i])
        return result

    def solve_branch_and_bound(vals, time_budget=20.0):
        """Branch and bound with bitsets for max sum-free subset."""
        n = len(vals)
        if n == 0:
            return []
        
        vals_set = set(vals)
        index_of_val = {v: i for i, v in enumerate(vals)}
        sum_mask = [0] * n
        double_mask = [0] * n
        
        for i in range(n):
            vi = vals[i]
            for j in range(n):
                s = vi + vals[j]
                if s in index_of_val:
                    k = index_of_val[s]
                    sum_mask[i] |= (1 << k)
            dv = 2 * vi
            if dv in index_of_val:
                k = index_of_val[dv]
                double_mask[i] |= (1 << k)
        
        def can_add_bitset(current_bits, idx):
            return (current_bits & sum_mask[idx]) == 0 and (current_bits & double_mask[idx]) == 0
        
        def count_bits(m):
            c = 0
            while m:
                c += 1
                m &= m - 1
            return c
        
        # Order elements by conflict count descending
        conflict_counts = []
        for i in range(n):
            mask = sum_mask[i] | double_mask[i]
            c = count_bits(mask)
            conflict_counts.append(c)
        
        order = sorted(range(n), key=lambda i: conflict_counts[i], reverse=True)
        
        new_vals = [vals[i] for i in order]
        old_to_new = {old_i: new_i for new_i, old_i in enumerate(order)}
        
        new_sum_mask = [0] * n
        new_double_mask = [0] * n
        for new_i, old_i in enumerate(order):
            old_mask = sum_mask[old_i]
            new_m = 0
            for old_j in range(n):
                if old_mask & (1 << old_j):
                    new_j = old_to_new[old_j]
                    new_m |= (1 << new_j)
            new_sum_mask[new_i] = new_m
            
            old_dmask = double_mask[old_i]
            new_dm = 0
            for old_j in range(n):
                if old_dmask & (1 << old_j):
                    new_j = old_to_new[old_j]
                    new_dm |= (1 << new_j)
            new_double_mask[new_i] = new_dm
        
        best_size = 0
        best_bits = 0
        end_time = time.time() + time_budget
        
        sys.setrecursionlimit(10000)
        
        def dfs(idx, current_bits):
            nonlocal best_size, best_bits
            
            if time.time() > end_time:
                return
            
            if idx == n:
                c = count_bits(current_bits)
                if c > best_size:
                    best_size = c
                    best_bits = current_bits
                return
            
            c = count_bits(current_bits)
            if c + (n - idx) <= best_size:
                return
            
            bit = 1 << idx
            if (current_bits & new_sum_mask[idx]) == 0 and (current_bits & new_double_mask[idx]) == 0:
                dfs(idx + 1, current_bits | bit)
            
            dfs(idx + 1, current_bits)
        
        dfs(0, 0)
        
        result = []
        for new_i in range(n):
            if best_bits & (1 << new_i):
                result.append(vals[order[new_i]])
        return result

    def solve_one(vals, time_budget=35.0):
        """Find a large sum-free subset of vals using multiple strategies."""
        n = len(vals)
        start = time.time()
        best = []
        best_size = 0
        rng = random.Random(42)
        
        # Strategy 1: Odd numbers + augment
        odd = [x for x in vals if x % 2 != 0]
        cand = greedy_augment(odd, vals)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
        
        # Strategy 2: Greedy from empty with various orderings
        orderings = [
            sorted(vals, key=lambda x: abs(x), reverse=True),
            sorted(vals, key=lambda x: abs(x)),
            sorted(vals, reverse=True),
            sorted(vals),
        ]
        
        for ordering in orderings:
            if time.time() - start > time_budget * 0.2:
                break
            cand = greedy_augment([], ordering)
            if len(cand) > best_size:
                best = cand
                best_size = len(cand)
        
        # Strategy 3: Local search on best
        remaining = time_budget - (time.time() - start)
        if remaining > 1.0:
            try:
                cand = local_search(best, vals, time_budget=min(remaining * 0.4, 5.0), rng=rng)
                if len(cand) > best_size:
                    best = cand
                    best_size = len(cand)
            except Exception:
                pass
        
        # Strategy 4: Simulated annealing
        remaining = time_budget - (time.time() - start)
        if remaining > 2.0:
            try:
                cand = solve_simulated_annealing(vals, time_budget=min(remaining * 0.5, 15.0), rng=rng)
                if len(cand) > best_size:
                    best = cand
                    best_size = len(cand)
            except Exception:
                pass
        
        # Strategy 5: Branch and bound (if time allows)
        remaining = time_budget - (time.time() - start)
        if remaining > 5.0 and n <= 150:
            try:
                cand = solve_branch_and_bound(vals, time_budget=min(remaining * 0.8, 20.0))
                if len(cand) > best_size:
                    best = cand
                    best_size = len(cand)
            except Exception:
                pass
        
        # Final local search on best
        remaining = time_budget - (time.time() - start)
        if remaining > 0.5:
            try:
                cand = local_search(best, vals, time_budget=remaining, rng=rng)
                if len(cand) > best_size:
                    best = cand
                    best_size = len(cand)
            except Exception:
                pass
        
        # Final safety check
        if not is_sum_free(best):
            best = odd
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
        
        vals = instance(size, seed)
        
        # Allocate time based on instance size
        if size <= 60:
            budget = 30.0
        elif size <= 120:
            budget = 40.0
        else:
            budget = 50.0
        
        remaining = DEADLINE - elapsed - 5
        budget = min(budget, remaining)
        
        result = solve_one(vals, time_budget=budget)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
