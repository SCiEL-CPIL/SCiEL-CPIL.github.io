def construct():
    import time
    from math import gcd
    from fractions import Fraction
    
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts: conflict[i] contains j if lcm(nums[i], nums[j]) <= 256
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            ai, aj = nums[i], nums[j]
            g = gcd(ai, aj)
            if ai * aj // g <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # We'll work with indices 0..254 representing nums 2..256
    # Key insight: small numbers (2-16) have high reciprocal value but create many conflicts.
    # Large numbers (17-256) have low reciprocal value but fewer conflicts.
    
    # Strategy: 
    # 1. Enumerate subsets of "critical" small numbers (2-16) that are independent.
    # 2. For each valid small subset, solve MWIS on remaining compatible large numbers.
    
    small_range = list(range(0, 15))  # indices for nums 2-16
    large_range = list(range(15, n))  # indices for nums 17-256
    
    best_sum = Fraction(0, 1)
    best_set = set()
    
    # Cache for large MWIS results
    large_cache = {}
    
    def solve_large_mwis(compatible_frozenset):
        """Solve MWIS on large numbers compatible with a given set of small numbers."""
        if compatible_frozenset in large_cache:
            return large_cache[compatible_frozenset]
        
        compatible_large = set(compatible_frozenset)
        if not compatible_large:
            large_cache[compatible_frozenset] = (set(), Fraction(0, 1))
            return set(), Fraction(0, 1)
        
        # Greedy initialization: sort by 1/x descending, add if no conflict
        candidates = sorted(compatible_large, key=lambda x: -1.0/nums[x])
        selected = set()
        sel_sum = Fraction(0, 1)
        for j in candidates:
            if not (conflict[j] & selected):
                selected.add(j)
                sel_sum += Fraction(1, nums[j])
        
        # Local search with multiple restarts
        for iteration in range(5):
            if time.time() > deadline:
                break
            
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                
                # Try adding any compatible node
                for j in compatible_large:
                    if j not in selected and not (conflict[j] & selected):
                        selected.add(j)
                        sel_sum += Fraction(1, nums[j])
                        improved = True
                
                # Try 1-for-1 swaps
                for i in list(selected):
                    s_without = selected - {i}
                    base_sum = sel_sum - Fraction(1, nums[i])
                    for j in compatible_large:
                        if j not in s_without and not (conflict[j] & s_without):
                            new_sum = base_sum + Fraction(1, nums[j])
                            if new_sum > sel_sum:
                                selected = s_without | {j}
                                sel_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
                
                # Try 2-for-1 swaps (remove 2, add 1)
                if not improved and len(selected) >= 2:
                    sel_list = list(selected)
                    for a_idx in range(len(sel_list)):
                        if time.time() > deadline:
                            break
                        for b_idx in range(a_idx+1, len(sel_list)):
                            i, j = sel_list[a_idx], sel_list[b_idx]
                            s_without = selected - {i, j}
                            base_sum = sel_sum - Fraction(1, nums[i]) - Fraction(1, nums[j])
                            for k in compatible_large:
                                if k not in s_without and not (conflict[k] & s_without):
                                    new_sum = base_sum + Fraction(1, nums[k])
                                    if new_sum > sel_sum:
                                        selected = s_without | {k}
                                        sel_sum = new_sum
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                
                # Try 1-for-2 swaps (remove 1, add 2)
                if not improved:
                    for i in list(selected):
                        if time.time() > deadline:
                            break
                        s_without = selected - {i}
                        base_sum = sel_sum - Fraction(1, nums[i])
                        # Find two nodes that can both be added
                        addable = [j for j in compatible_large if j not in s_without and not (conflict[j] & s_without)]
                        for a_idx in range(len(addable)):
                            for b_idx in range(a_idx+1, len(addable)):
                                j, k = addable[a_idx], addable[b_idx]
                                if not (conflict[j] & conflict[k]) and not (conflict[j] & s_without) and not (conflict[k] & s_without):
                                    new_sum = base_sum + Fraction(1, nums[j]) + Fraction(1, nums[k])
                                    if new_sum > sel_sum:
                                        selected = s_without | {j, k}
                                        sel_sum = new_sum
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
        
        result = (selected, sel_sum)
        large_cache[compatible_frozenset] = result
        return selected, sel_sum
    
    # Enumerate subsets of small numbers
    # But 2^15 = 32768 is manageable
    small_count = len(small_range)
    
    for mask in range(1 << small_count):
        if time.time() > deadline:
            break
        
        s_small = set()
        for i in range(small_count):
            if mask & (1 << i):
                s_small.add(small_range[i])
        
        # Check independence of small subset
        valid = True
        for i in s_small:
            if conflict[i] & s_small:
                valid = False
                break
        if not valid:
            continue
        
        # Find compatible large numbers
        compatible_large = frozenset(j for j in large_range if not (conflict[j] & s_small))
        
        large_set, large_sum = solve_large_mwis(compatible_large)
        
        small_sum = sum(Fraction(1, nums[x]) for x in s_small)
        total_sum = small_sum + large_sum
        
        if total_sum > best_sum:
            best_sum = total_sum
            best_set = s_small | large_set
    
    # Convert to list of actual numbers
    result = sorted(nums[i] for i in best_set)
    if not result:
        return [2]
    return result
