"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

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
        """Check if adding x to S keeps it sum-free."""
        if 2 * x in S_set:
            return False
        for a in S:
            if a + x in S_set:
                return False
        return True

    def greedy_augment(S, all_elements):
        """Try to add elements from all_elements not in S, in a given order."""
        S_set = set(S)
        S_list = list(S)
        for x in all_elements:
            if x not in S_set:
                if can_add(S_list, S_set, x):
                    S_set.add(x)
                    S_list.append(x)
        return S_list

    def local_search(S, all_elements, time_budget=30.0):
        """Hill-climbing with 1-swap moves to escape local optima."""
        S_set = set(S)
        S_list = list(S)
        best_size = len(S_list)
        all_set = set(all_elements)
        start = time.time()
        
        while time.time() - start < time_budget:
            improved = False
            for x in list(S_list):
                if time.time() - start >= time_budget:
                    break
                # Try removing x and adding any y not in S
                temp_set = S_set - {x}
                for y in all_set - S_set:
                    if y == x:
                        continue
                    if can_add(list(temp_set), temp_set, y):
                        new_set = temp_set | {y}
                        # Augment after swap
                        aug_list = greedy_augment(list(new_set), all_elements)
                        if len(aug_list) > best_size:
                            S_list = aug_list
                            S_set = set(aug_list)
                            best_size = len(S_list)
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return S_list

    def solve_one(vals, time_budget=35.0):
        """Find a large sum-free subset of vals using multiple strategies."""
        n = len(vals)
        start = time.time()
        best = []
        best_size = 0
        
        # Strategy 1: Odd numbers (always sum-free) + augment
        odd = [x for x in vals if x % 2 != 0]
        cand = greedy_augment(odd, vals)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
        
        # Strategy 2: Greedy from empty, sorted by various orderings
        orderings = [
            sorted(vals, key=lambda x: abs(x), reverse=True),  # largest abs first
            sorted(vals, key=lambda x: abs(x)),                  # smallest abs first
            sorted(vals, reverse=True),                          # largest value first
            sorted(vals),                                        # smallest value first
            sorted(vals, key=lambda x: x % 2, reverse=True),     # odds first
        ]
        
        for ordering in orderings:
            if time.time() - start > time_budget * 0.4:
                break
            cand = greedy_augment([], ordering)
            if len(cand) > best_size:
                best = cand
                best_size = len(cand)
        
        # Strategy 3: Positive-only subset
        positives = [x for x in vals if x > 0]
        if positives:
            cand = greedy_augment(positives, vals)
            if len(cand) > best_size:
                best = cand
                best_size = len(cand)
        
        # Strategy 4: Negative-only subset
        negatives = [x for x in vals if x < 0]
        if negatives:
            cand = greedy_augment(negatives, vals)
            if len(cand) > best_size:
                best = cand
                best_size = len(cand)
        
        # Local search to improve the best candidate
        remaining = time_budget - (time.time() - start)
        if remaining > 2.0 and best_size < n:
            improved = local_search(best, vals, time_budget=remaining - 1.0)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
        
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
            budget = 25.0
        elif size <= 120:
            budget = 35.0
        else:
            budget = 45.0
        
        # Ensure we don't exceed deadline
        remaining = DEADLINE - elapsed - 5
        budget = min(budget, remaining)
        
        result = solve_one(vals, time_budget=budget)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
