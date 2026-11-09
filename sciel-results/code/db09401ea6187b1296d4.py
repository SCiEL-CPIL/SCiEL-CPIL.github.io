import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s
    n_max = 512
    target_size = 48

    # Precompute conflict degrees and a fast lookup
    conflict_degree = [0] * (n_max + 1)
    for x in range(1, n_max + 1):
        a_min = max(1, 2 * x - n_max)
        a_max = x - 1
        if a_min <= a_max:
            conflict_degree[x] = a_max - a_min + 1

    # Precompute sorted orders
    ordered_by_degree_asc = sorted(range(1, n_max + 1), key=lambda x: conflict_degree[x])
    ordered_by_degree_desc = sorted(range(1, n_max + 1), key=lambda x: -conflict_degree[x])

    # Precompute pair conflicts: for each x, which y values conflict with x (x is mean of {y1,y2} or y is mean of {x,y2})
    # x conflicts with y if there exists z != x,y such that x = (y+z)/2 or y = (x+z)/2
    # i.e., 2x = y + z or 2y = x + z for some z in [1,n_max], z != x, z != y
    # For fixed x,y: conflict if 2x-y is in [1,n_max] and 2x-y != x and 2x-y != y
    #                 or 2y-x is in [1,n_max] and 2y-x != x and 2y-x != y
    pair_conflict = [set() for _ in range(n_max + 1)]
    for x in range(1, n_max + 1):
        for y in range(1, n_max + 1):
            if x == y:
                continue
            # Check if x = (y + z) / 2 for some z != x, z != y
            z = 2 * x - y
            if 1 <= z <= n_max and z != x and z != y:
                pair_conflict[x].add(y)
                pair_conflict[y].add(x)
                continue
            # Check if y = (x + z) / 2 for some z != x, z != y
            z = 2 * y - x
            if 1 <= z <= n_max and z != x and z != y:
                pair_conflict[x].add(y)
                pair_conflict[y].add(x)

    def check_incremental(S_list, x):
        """
        Check if adding x to set S_list (where S_list is already valid) keeps the set valid.
        Only checks constraints that involve x.
        Returns True if S_list U {x} is valid.
        """
        if x in S_list:
            return False
        
        n = len(S_list)
        
        # Quick pair conflict check
        for s in S_list:
            if s in pair_conflict[x]:
                return False
        
        # Check 1: x is not the mean of any subset of S_list of size >= 2
        # sum(T) = k * x for some T subset of S_list, |T| = k >= 2
        if n >= 2:
            max_sum = sum(S_list)
            max_k = min(n, max_sum // x)
            if max_k >= 2:
                dp = [0] * (n + 1)
                dp[0] = 1
                for val in S_list:
                    for k in range(n - 1, -1, -1):
                        if dp[k]:
                            dp[k+1] |= (dp[k] << val)
                for k in range(2, max_k + 1):
                    target = k * x
                    if (dp[k] >> target) & 1:
                        return False
        
        # Check 2: No element a in S_list is the mean of any subset of (S_list \ {a}) U {x} of size >= 2
        # Only need to check subsets that include x (subsets without x were already valid)
        for a in S_list:
            others = [s for s in S_list if s != a]
            m = len(others)
            if m < 1:
                continue
            
            # Quick check: if x and a form a pair conflict with some other element, we already caught it
            # But we still need to check larger subsets
            
            # Subset M = {x} U T, T subset of others, |T| >= 1
            # sum(M) = x + sum(T) = (|T| + 1) * a
            # sum(T) = (j + 1) * a - x where j = |T|, j >= 1
            dp = [0] * (m + 1)
            dp[0] = 1
            for val in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << val)
            
            max_sum_o = sum(others)
            for j in range(1, m + 1):
                target = (j + 1) * a - x
                if target < 0:
                    continue
                if target > max_sum_o:
                    break
                if (dp[j] >> target) & 1:
                    return False
        
        return True

    def check_full_validity(S_list, x):
        """
        Full validity check for use in local search when S_list may not be fully valid.
        """
        if x in S_list:
            return False
        
        n = len(S_list)
        
        # Quick pair conflict check
        for s in S_list:
            if s in pair_conflict[x]:
                return False
        
        # Check 1: x is not the mean of any subset of S_list of size >= 2
        if n >= 2:
            max_sum = sum(S_list)
            max_k = min(n, max_sum // x)
            if max_k >= 2:
                dp = [0] * (n + 1)
                dp[0] = 1
                for val in S_list:
                    for k in range(n - 1, -1, -1):
                        if dp[k]:
                            dp[k+1] |= (dp[k] << val)
                for k in range(2, max_k + 1):
                    target = k * x
                    if (dp[k] >> target) & 1:
                        return False
        
        # Check 2: No element a in S_list is the mean of any subset of (S_list \ {a}) U {x} of size >= 2
        for a in S_list:
            others = [s for s in S_list if s != a]
            m = len(others)
            if m < 1:
                continue
            
            dp = [0] * (m + 1)
            dp[0] = 1
            for val in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << val)
            
            max_sum_o = sum(others)
            for j in range(1, m + 1):
                target = (j + 1) * a - x
                if target < 0:
                    continue
                if target > max_sum_o:
                    break
                if (dp[j] >> target) & 1:
                    return False
        
        return True

    def greedy_conflict_aware(order, S_set):
        """
        Greedy independent set with conflict-aware ordering.
        """
        S = []
        S_set = set(S_set) if S_set else set()
        
        for x in order:
            if len(S) >= target_size:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if check_incremental(S, x):
                S.append(x)
                S_set.add(x)
        
        return S, S_set

    def local_search(S, S_set, available, rng, max_iters=200):
        """
        Local search: try to add or swap elements to improve the set.
        """
        S = list(S)
        S_set = set(S)
        available = list(available)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline:
            iter_count += 1
            improved = False
            rng.shuffle(available)
            
            # Try to add any available element
            for x in available:
                if len(S) >= target_size:
                    break
                if check_incremental(S, x):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            
            if len(S) >= target_size:
                break
            
            # If no add, try swaps
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    old = S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_temp_set = set(S_temp)
                    
                    for x in available:
                        if x in S_temp_set:
                            continue
                        if check_full_validity(S_temp, x):
                            S = S_temp + [x]
                            S_set = set(S)
                            available.remove(x)
                            available.append(old)
                            improved = True
                            break
                    if improved:
                        break
        
        return S, S_set, available

    def polish_pass(S, S_set, rng, max_iters=50):
        """
        Final polish: try targeted swaps to fill remaining slots.
        Focus on replacing high-degree elements with lower-degree candidates.
        """
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, n_max + 1) if x not in S_set]
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline:
            iter_count += 1
            improved = False
            
            if len(S) >= target_size:
                break
            
            # Sort S by conflict degree descending (worst first for replacement)
            S_sorted = sorted(range(len(S)), key=lambda i: -conflict_degree[S[i]])
            
            for idx in S_sorted:
                if time.time() > deadline:
                    break
                if len(S) >= target_size:
                    break
                
                old = S[idx]
                S_temp = S[:idx] + S[idx+1:]
                S_temp_set = set(S_temp)
                
                # Try replacing with lower-degree available elements
                candidates = sorted(available, key=lambda x: conflict_degree[x])
                for x in candidates:
                    if x in S_temp_set:
                        continue
                    if check_full_validity(S_temp, x):
                        S = S_temp + [x]
                        S_set = set(S)
                        available.remove(x)
                        available.append(old)
                        improved = True
                        break
                if improved:
                    break
        
        return S, S_set, available

    # Strategy 1: Greedy with conflict-aware ordering (lowest degree first)
    best_set = []
    best_size = 0
    
    S, S_set = greedy_conflict_aware(ordered_by_degree_asc, set())
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Strategy 2: Descending degree
    S, S_set = greedy_conflict_aware(ordered_by_degree_desc, set())
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Strategy 3: Multiple random orders with different seeds
    rng = random.Random(42)
    num_random_trials = 80
    for i in range(num_random_trials):
        if time.time() > deadline:
            break
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        S, S_set = greedy_conflict_aware(order, set())
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Strategy 4: Weighted random orders biased toward low-conflict elements
    for i in range(30):
        if time.time() > deadline:
            break
        weights = [1.0 / (1 + conflict_degree[x]) for x in range(1, n_max + 1)]
        order = []
        remaining = list(range(1, n_max + 1))
        w = list(weights)
        while remaining:
            total_w = sum(w)
            r = rng.random() * total_w
            cum = 0
            for j, val in enumerate(remaining):
                cum += w[j]
                if cum >= r:
                    order.append(val)
                    remaining.pop(j)
                    w.pop(j)
                    break
        S, S_set = greedy_conflict_aware(order, set())
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Local search phase with multiple restarts
    if best_size < target_size and time.time() < deadline:
        S = list(best_set)
        S_set = set(S)
        available = [x for x in range(1, n_max + 1) if x not in S_set]
        
        # Try local search with different random seeds
        for seed in range(20):
            if time.time() > deadline:
                break
            rng_ls = random.Random(seed * 1000 + 7)
            S_ls, S_set_ls, available_ls = local_search(S, S_set, available, rng_ls, max_iters=100)
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_set = list(S_ls)
                S = list(S_ls)
                S_set = set(S_ls)
                available = [x for x in range(1, n_max + 1) if x not in S_set]
    
    # Final polish pass
    if best_size < target_size and time.time() < deadline:
        S = list(best_set)
        S_set = set(S)
        rng_polish = random.Random(9999)
        S_polished, S_set_polished, _ = polish_pass(S, S_set, rng_polish, max_iters=80)
        if len(S_polished) > best_size:
            best_size = len(S_polished)
            best_set = list(S_polished)
    
    return best_set
