"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    MASK = (1 << 201) - 1
    TARGET_BIT = 1 << 200
    
    def greedy(order):
        """Greedy: include numbers in given order if it doesn't create sum 200.
        Uses bitset for speed."""
        reachable = 1  # bit 0 set
        selected = []
        for x in order:
            new_bits = (reachable << x) & MASK
            if new_bits & TARGET_BIT:
                continue
            reachable |= new_bits
            selected.append(x)
        return selected
    
    def compute_reachable(lst):
        """Compute reachable bitset for a list."""
        r = 1
        for x in lst:
            r |= (r << x) & MASK
        return r
    
    def compute_prefix_suffix(lst):
        """Compute prefix and suffix reachable bitsets for fast removal."""
        n = len(lst)
        prefix = [1] * (n + 1)
        suffix = [1] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] | ((prefix[i] << lst[i]) & MASK)
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i+1] | ((suffix[i+1] << lst[i]) & MASK)
        return prefix, suffix
    
    def reachable_without(lst, idx):
        """Get reachable bitset with element at idx removed, using prefix/suffix."""
        prefix, suffix = compute_prefix_suffix(lst)
        # Combine prefix up to idx and suffix from idx+1
        # We need to merge prefix[idx] and suffix[idx+1]
        # But they're not simply OR-able since suffix depends on elements after idx
        # Actually we need to recompute: take prefix[idx] and then add elements idx+1..n-1
        # This is still O(n) but avoids full recompute in some cases
        # For now, just recompute - the prefix/suffix approach needs more thought
        result = 1
        for i in range(len(lst)):
            if i != idx:
                result |= (result << lst[i]) & MASK
        return result
    
    def local_improve_fast(selected, excluded):
        """Fast local improvement using bitset operations."""
        best = list(selected)
        excluded_list = list(excluded)
        best_set = set(best)
        reachable = compute_reachable(best)
        
        improved = True
        max_iterations = 500
        iteration = 0
        while improved and time.time() < deadline and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Phase 1: Try adding each excluded element directly
            added = False
            for i, x in enumerate(excluded_list):
                new_bits = (reachable << x) & MASK
                if not (new_bits & TARGET_BIT):
                    best.append(x)
                    best_set.add(x)
                    excluded_list.pop(i)
                    reachable |= new_bits
                    improved = True
                    added = True
                    break
            if added:
                continue
            
            # Phase 2: Try swap remove-1-add-1
            swapped = False
            for i, y in enumerate(best):
                # Quick check: if y is large, removing it might help
                best_wo_y = [z for k, z in enumerate(best) if k != i]
                reachable_no_y = compute_reachable(best_wo_y)
                for j, x in enumerate(excluded_list):
                    if x == y:
                        continue
                    new_bits = (reachable_no_y << x) & MASK
                    if not (new_bits & TARGET_BIT):
                        best[i] = x
                        best_set.discard(y)
                        best_set.add(x)
                        excluded_list[j] = y
                        reachable = compute_reachable(best)
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break
            if swapped:
                continue
            
            # Phase 3: Try swap remove-1-add-2 (net gain of 1)
            n_best = len(best)
            for i in range(n_best):
                y = best[i]
                best_wo_y = [z for k, z in enumerate(best) if k != i]
                reachable_no_y = compute_reachable(best_wo_y)
                found = False
                for a_idx in range(len(excluded_list)):
                    x1 = excluded_list[a_idx]
                    b1 = (reachable_no_y << x1) & MASK
                    if b1 & TARGET_BIT:
                        continue
                    for b_idx in range(len(excluded_list)):
                        if b_idx == a_idx:
                            continue
                        x2 = excluded_list[b_idx]
                        b2 = (b1 << x2) & MASK
                        if b2 & TARGET_BIT:
                            continue
                        # Perform swap
                        new_best = best_wo_y + [x1, x2]
                        new_excluded = []
                        for k, ex in enumerate(excluded_list):
                            if k == a_idx or k == b_idx:
                                continue
                            new_excluded.append(ex)
                        new_excluded.append(y)
                        
                        best = new_best
                        best_set = set(best)
                        excluded_list = new_excluded
                        reachable = compute_reachable(best)
                        improved = True
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if improved:
                continue
        
        return best
    
    # Try multiple orderings
    all_numbers = list(range(1, 101))
    best_result = []
    
    # Descending order (usually best for this type of problem)
    r = greedy(all_numbers[::-1])
    if len(r) > len(best_result):
        best_result = r
    
    # Ascending order
    r = greedy(all_numbers)
    if len(r) > len(best_result):
        best_result = r
    
    # Try ordering by value descending, but skip values that are exactly half of 200
    # Actually, let's try several smart orderings
    # Order: large values first (they're "harder" to combine to make 200)
    for mod in [3, 4, 5, 6, 7]:
        groups = [[] for _ in range(mod)]
        for x in all_numbers:
            groups[x % mod].append(x)
        # Sort each group descending, then interleave
        for g in groups:
            g.sort(reverse=True)
        order = []
        max_len = max(len(g) for g in groups)
        for i in range(max_len):
            for g in groups:
                if i < len(g):
                    order.append(g[i])
        r = greedy(order)
        if len(r) > len(best_result):
            best_result = r
    
    # Random shuffles with limited time
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 20 and attempts < 50:
        order = all_numbers[:]
        rng.shuffle(order)
        r = greedy(order)
        if len(r) > len(best_result):
            best_result = r
            attempts = 0
        else:
            attempts += 1
    
    # Local improvement on best result
    selected_set = set(best_result)
    excluded = [x for x in all_numbers if x not in selected_set]
    best_result = local_improve_fast(best_result, excluded)
    
    # Try local improvement from descending greedy too
    r_desc = greedy(all_numbers[::-1])
    if len(r_desc) >= len(best_result) - 3:
        sel_set = set(r_desc)
        exc = [x for x in all_numbers if x not in sel_set]
        r_improved = local_improve_fast(r_desc, exc)
        if len(r_improved) > len(best_result):
            best_result = r_improved
    
    if not best_result:
        return [100]
    
    return best_result
# EVOLVE-BLOCK-END
