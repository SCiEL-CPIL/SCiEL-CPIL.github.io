import time
from math import gcd
from fractions import Fraction
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Range [2, 256]
    nums = list(range(2, 257))
    n = len(nums)
    num_to_idx = {v: i for i, v in enumerate(nums)}
    
    # Build conflict graph: conflict[i] is set of indices j where lcm(nums[i], nums[j]) <= 256
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _lcm(nums[i], nums[j]) <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    if time.time() - start_time > deadline:
        return [2]
    
    weights = [1.0 / x for x in nums]
    
    # Start with the known valid set: {129, 130, ..., 256}
    # These are indices where nums[i] >= 129
    initial_set = set(i for i in range(n) if nums[i] >= 129)
    current_weight = sum(weights[i] for i in initial_set)
    best_set = set(initial_set)
    best_weight = current_weight
    
    def is_valid_set(s):
        """Check if set s is independent (no conflicts)."""
        for i in s:
            for j in s:
                if i != j and j in conflict[i]:
                    return False
        return True
    
    # Local search: try adding a single vertex
    def try_add_single(s, w):
        """Try to add a single vertex not in s that doesn't conflict with s."""
        for i in range(n):
            if i in s:
                continue
            # Check if i conflicts with any in s
            if conflict[i] & s:
                continue
            # Can add i
            new_s = s | {i}
            new_w = w + weights[i]
            return new_s, new_w, True
        return s, w, False
    
    # Try removing one vertex and adding multiple
    def try_remove_add(s, w, max_adds=3):
        """Try removing one vertex from s and adding up to max_adds new vertices."""
        best_local_s = s
        best_local_w = w
        for rem in list(s):
            s_without_rem = s - {rem}
            w_without_rem = w - weights[rem]
            # Find all vertices that can be added to s_without_rem
            candidates = []
            for i in range(n):
                if i in s_without_rem:
                    continue
                if conflict[i] & s_without_rem:
                    continue
                candidates.append(i)
            # Sort by weight descending
            candidates.sort(key=lambda i: -weights[i])
            # Greedily add up to max_adds
            new_s = set(s_without_rem)
            new_w = w_without_rem
            for i in candidates:
                if len(new_s - s_without_rem) >= max_adds:
                    break
                if i in new_s:
                    continue
                if conflict[i] & new_s:
                    continue
                new_s.add(i)
                new_w += weights[i]
            if new_w > best_local_w:
                best_local_w = new_w
                best_local_s = set(new_s)
        return best_local_s, best_local_w
    
    # Iterate local search
    improved = True
    iteration = 0
    while improved and time.time() - start_time < deadline:
        improved = False
        iteration += 1
        
        # Try adding single vertices
        new_s, new_w, added = try_add_single(best_set, best_weight)
        if added and new_w > best_weight:
            best_set = new_s
            best_weight = new_w
            improved = True
            continue
        
        # Try remove-one-add-multiple
        new_s, new_w = try_remove_add(best_set, best_weight, max_adds=4)
        if new_w > best_weight + 1e-12:
            best_set = new_s
            best_weight = new_w
            improved = True
            continue
        
        # Try 2-swap: remove two, add up to 4
        if iteration % 5 == 0:  # do this less frequently
            for rem1, rem2 in combinations(list(best_set), 2):
                s_without = best_set - {rem1, rem2}
                w_without = best_weight - weights[rem1] - weights[rem2]
                candidates = []
                for i in range(n):
                    if i in s_without:
                        continue
                    if conflict[i] & s_without:
                        continue
                    candidates.append(i)
                candidates.sort(key=lambda i: -weights[i])
                new_s = set(s_without)
                new_w = w_without
                for i in candidates:
                    if len(new_s - s_without) >= 4:
                        break
                    if i in new_s:
                        continue
                    if conflict[i] & new_s:
                        continue
                    new_s.add(i)
                    new_w += weights[i]
                if new_w > best_weight + 1e-12:
                    best_set = new_s
                    best_weight = new_w
                    improved = True
                    break
            if improved:
                continue
        
        if time.time() - start_time > deadline:
            break
    
    # Also try a greedy from scratch as alternative
    order = sorted(range(n), key=lambda i: -weights[i])
    greedy_set = set()
    greedy_weight = 0.0
    for idx in order:
        if not (conflict[idx] & greedy_set):
            greedy_set.add(idx)
            greedy_weight += weights[idx]
    
    if greedy_weight > best_weight:
        best_set = greedy_set
        best_weight = greedy_weight
    
    # Final verification
    result = [nums[i] for i in best_set]
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if _lcm(result[i], result[j]) <= 256:
                # Fall back to safe set
                return [nums[i] for i in initial_set]
    
    if not result:
        return [256]
    
    return result
