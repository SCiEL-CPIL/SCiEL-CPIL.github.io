"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 88  # Safety margin for 90s limit

    # Generate smooth numbers (2^i * 3^j) as primary candidates
    smooth = set()
    for i in range(10):
        for j in range(7):
            v = (1 << i) * (3 ** j)
            if 1 <= v <= 512:
                smooth.add(v)

    all_candidates = list(range(1, 513))
    max_n = 512

    def is_valid_set(selected):
        """Check if a set is valid: no element is the mean of any subset of >=2 other elements."""
        n = len(selected)
        if n < 3:
            return True
        
        # For each element a, check if it's the mean of any subset of others
        # a = sum(S)/k where S is a subset of selected\{a} with |S| = k >= 2
        # => sum(S) = k*a
        
        for idx, a in enumerate(selected):
            others = [selected[i] for i in range(n) if i != idx]
            m = len(others)
            if m < 2:
                continue
            
            # DP: dp[k] = bitset of sums achievable with exactly k elements from others
            dp = [0] * (m + 1)
            dp[0] = 1
            for elem in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k - 1] << elem)
            
            for k in range(2, m + 1):
                target = k * a
                if (dp[k] >> target) & 1:
                    return False
        
        return True

    def can_add(selected, x):
        """Check if x can be added to selected without violating constraints."""
        n = len(selected)
        
        # Check 1: Is x the mean of any subset of >=2 elements from selected?
        if n >= 2:
            dp = [0] * (n + 1)
            dp[0] = 1
            for elem in selected:
                for k in range(n, 0, -1):
                    dp[k] |= (dp[k - 1] << elem)
            
            max_sum = sum(selected)
            for k in range(2, n + 1):
                target = k * x
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        
        # Check 2: Does adding x make any a in selected the mean of a subset
        #         of >=2 elements from (selected \ {a}) ∪ {x}?
        for a in selected:
            others = [y for y in selected if y != a]
            m = len(others)
            if m < 1:
                continue
            
            # We need: sum(S) + x = (j+1) * a for some S subset of others, |S| = j >= 1
            # => sum(S) = (j+1)*a - x
            dp2 = [0] * (m + 1)
            dp2[0] = 1
            for elem in others:
                for k in range(m, 0, -1):
                    dp2[k] |= (dp2[k - 1] << elem)
            
            max_others = sum(others)
            for j in range(1, m + 1):
                target = (j + 1) * a - x
                if target < 0:
                    continue
                if target > max_others:
                    break
                if (dp2[j] >> target) & 1:
                    return False
        
        return True

    def greedy_construct(order, selected=None):
        """Greedy construction with given order."""
        if selected is None:
            selected = []
        selected_set = set(selected)
        
        for x in order:
            if time.time() > deadline:
                break
            if x in selected_set:
                continue
            
            if can_add(selected, x):
                selected.append(x)
                selected_set.add(x)
                if len(selected) >= 48:
                    break
        
        return selected

    def local_improve(selected, budget):
        """Try to improve the set by swapping elements."""
        selected = list(selected)
        selected_set = set(selected)
        n = len(selected)
        
        # Try removing one element and adding a better one
        for i in range(n):
            if time.time() > budget:
                break
            
            a = selected[i]
            temp_selected = [x for x in selected if x != a]
            
            # Find best replacement
            best_replacement = None
            best_score = -1
            
            for x in range(1, 513):
                if x in selected_set and x != a:
                    continue
                
                if can_add(temp_selected, x):
                    # Score: prefer numbers that are "far" from existing elements
                    # to reduce future conflicts
                    min_dist = min(abs(x - y) for y in temp_selected) if temp_selected else 512
                    score = min_dist
                    
                    if score > best_score:
                        best_score = score
                        best_replacement = x
            
            if best_replacement is not None:
                new_selected = temp_selected + [best_replacement]
                new_selected.sort()
                if is_valid_set(new_selected):
                    selected = new_selected
                    selected_set = set(selected)
        
        return selected

    # Strategy 1: Smooth numbers first
    smooth_list = sorted(smooth)
    rest = [x for x in all_candidates if x not in smooth]
    order1 = smooth_list + sorted(rest)
    
    best = greedy_construct(order1)
    
    # Strategy 2: Descending order
    if len(best) < 48 and time.time() < deadline - 15:
        order2 = sorted(all_candidates, reverse=True)
        result2 = greedy_construct(order2)
        if len(result2) > len(best):
            best = result2
    
    # Strategy 3: Randomized greedy with smooth bias
    if len(best) < 48 and time.time() < deadline - 15:
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Create a randomized order with smooth numbers weighted higher
            order3 = list(all_candidates)
            random.shuffle(order3)
            
            # Weight smooth numbers to appear earlier
            smooth_weighted = [x for x in order3 if x in smooth]
            non_smooth_weighted = [x for x in order3 if x not in smooth]
            
            # Interleave with smooth bias
            order3_final = []
            i, j = 0, 0
            while i < len(smooth_weighted) or j < len(non_smooth_weighted):
                # Add 2 smooth for every 1 non-smooth on average
                if i < len(smooth_weighted) and (j >= len(non_smooth_weighted) or random.random() < 0.6):
                    order3_final.append(smooth_weighted[i])
                    i += 1
                elif j < len(non_smooth_weighted):
                    order3_final.append(non_smooth_weighted[j])
                    j += 1
                elif i < len(smooth_weighted):
                    order3_final.append(smooth_weighted[i])
                    i += 1
            
            result3 = greedy_construct(order3_final)
            if len(result3) > len(best):
                best = result3
    
    # Strategy 4: Local improvement on best so far
    if time.time() < deadline - 10:
        improved = local_improve(best, deadline - 5)
        if len(improved) > len(best):
            best = improved
    
    # Strategy 5: Try to fill gaps - find missing numbers that can be added
    if len(best) < 48 and time.time() < deadline - 5:
        best_set = set(best)
        best_sorted = sorted(best)
        
        for x in range(1, 513):
            if time.time() > deadline - 3:
                break
            if x in best_set:
                continue
            
            if can_add(best_sorted, x):
                best_sorted.append(x)
                best_set.add(x)
                best_sorted.sort()
                if len(best_sorted) >= 48:
                    break
        
        best = best_sorted
    
    return best

# EVOLVE-BLOCK-END
