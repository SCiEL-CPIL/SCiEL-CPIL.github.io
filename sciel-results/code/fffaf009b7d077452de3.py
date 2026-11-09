"""Direct construction exploiting divisibility structure of the LCM constraint."""
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)

    # Precompute conflict masks using bitsets for speed
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Precompute compatibility: comp[i] = mask of elements compatible with i
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = ((1 << n) - 1) & ~(conflict_mask[i] | (1 << i))

    weights = [1.0 / numbers[i] for i in range(n)]
    # Also keep exact fractions for tie-breaking
    exact_weights = [Fraction(1, numbers[i]) for i in range(n)]

    def reciprocal_sum_from_mask(mask):
        total = 0.0
        for i in range(n):
            if mask & (1 << i):
                total += weights[i]
        return total

    def can_add(i, selected_mask):
        return (conflict_mask[i] & selected_mask) == 0

    # === DIRECT CONSTRUCTION ===
    # Key structural facts:
    # 1. If 2 in S, no other even can be (lcm(2,x)=x<=256 for even x)
    # 2. Odds > 128 are mutually compatible (lcm > 256 always)
    # 3. Evens don't conflict with odds > 128 (lcm(e,o) > 256 for even e>=2, odd o>128)

    # Build the direct solution: {2} union {odd numbers in (128, 256]}
    selected_mask = 0
    for i in range(n):
        val = numbers[i]
        if val == 2:
            selected_mask |= (1 << i)
        elif val % 2 == 1 and val > 128:
            selected_mask |= (1 << i)
    
    best_mask = selected_mask
    best_sum = reciprocal_sum_from_mask(selected_mask)
    
    # Verify validity
    def is_valid(mask):
        for i in range(n):
            if mask & (1 << i):
                for j in range(i+1, n):
                    if mask & (1 << j):
                        if conflict_mask[i] & (1 << j):
                            return False
        return True
    
    assert is_valid(best_mask), "Direct construction is invalid!"
    
    # === GREEDY FROM MULTIPLE STARTS ===
    def greedy(order):
        mask = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)
        return mask
    
    # Ordering 1: weight descending
    order_w = list(range(n))
    order_w.sort(key=lambda i: -weights[i])
    mask_w = greedy(order_w)
    sum_w = reciprocal_sum_from_mask(mask_w)
    if sum_w > best_sum:
        best_sum = sum_w
        best_mask = mask_w
    
    # Ordering 2: "no 2" strategy - exclude 2, prioritize evens by weight
    order_no2 = [i for i in range(n) if numbers[i] != 2]
    order_no2.sort(key=lambda i: -weights[i])
    mask_no2 = greedy(order_no2)
    sum_no2 = reciprocal_sum_from_mask(mask_no2)
    if sum_no2 > best_sum:
        best_sum = sum_no2
        best_mask = mask_no2
    
    # Ordering 3: "no 2" with evens first
    order_ev_first = [i for i in range(n) if numbers[i] != 2 and numbers[i] % 2 == 0]
    order_odd_first = [i for i in range(n) if numbers[i] != 2 and numbers[i] % 2 == 1]
    order_ev_first.sort(key=lambda i: -weights[i])
    order_odd_first.sort(key=lambda i: -weights[i])
    mask_mixed = greedy(order_ev_first + order_odd_first)
    sum_mixed = reciprocal_sum_from_mask(mask_mixed)
    if sum_mixed > best_sum:
        best_sum = sum_mixed
        best_mask = mask_mixed
    
    # Ordering 4: Anchor small evens (4,6,8,...) and fill rest
    for anchor_val in [4, 6, 8, 10, 12]:
        anchor_idx = numbers.index(anchor_val) if anchor_val in numbers else -1
        if anchor_idx < 0:
            continue
        anchor_mask = 1 << anchor_idx
        # Greedy fill excluding conflicts with anchor
        order_anchor = [i for i in range(n) if i != anchor_idx]
        order_anchor.sort(key=lambda i: -weights[i])
        m = anchor_mask
        for i in order_anchor:
            if can_add(i, m):
                m |= (1 << i)
        s = reciprocal_sum_from_mask(m)
        if s > best_sum:
            best_sum = s
            best_mask = m
    
    # === LOCAL SEARCH ===
    def try_additions(mask):
        new_mask = mask
        for i in range(n):
            if not (new_mask & (1 << i)):
                if can_add(i, new_mask):
                    new_mask |= (1 << i)
        return new_mask
    
    def local_search(mask, max_iters=100):
        """Iterative local search with 1-swap and 2-swap moves using precomputed compatibility."""
        current_mask = mask
        current_sum = reciprocal_sum_from_mask(current_mask)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE:
            improved = False
            iters += 1
            
            # Phase 1: Add all possible
            current_mask = try_additions(current_mask)
            current_sum = reciprocal_sum_from_mask(current_mask)
            
            # Phase 2: Try 1-removal + greedy fill
            found_improvement = False
            for i in range(n):
                if elapsed() > DEADLINE:
                    break
                if not (current_mask & (1 << i)):
                    continue
                mask_no_i = current_mask & ~(1 << i)
                new_mask = try_additions(mask_no_i)
                new_s = reciprocal_sum_from_mask(new_mask)
                if new_s > current_sum + 1e-12:
                    current_mask = new_mask
                    current_sum = new_s
                    improved = True
                    found_improvement = True
                    break
            
            # Phase 3: Try 2-swap (remove one, add another) - optimized
            if not improved and iters < max_iters and elapsed() < DEADLINE:
                best_swap_gain = 0.0
                best_i = -1
                best_j = -1
                for i in range(n):
                    if elapsed() > DEADLINE:
                        break
                    if not (current_mask & (1 << i)):
                        continue
                    mask_no_i = current_mask & ~(1 << i)
                    # Find best j to add
                    for j in range(n):
                        if (current_mask & (1 << j)):
                            continue
                        if can_add(j, mask_no_i):
                            gain = weights[j] - weights[i]
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_i = i
                                best_j = j
                if best_i >= 0 and best_swap_gain > 1e-12:
                    new_mask = (current_mask & ~(1 << best_i)) | (1 << best_j)
                    current_mask = new_mask
                    current_sum += best_swap_gain
                    improved = True
            
            # Phase 4: Try 2-removal + 2-addition (2-swap)
            if not improved and iters < max_iters and elapsed() < DEADLINE:
                best_2swap_gain = 0.0
                best_i1 = -1
                best_i2 = -1
                best_j1 = -1
                best_j2 = -1
                selected_indices = [i for i in range(n) if current_mask & (1 << i)]
                unselected_indices = [i for i in range(n) if not (current_mask & (1 << i))]
                
                # Try pairs of removals
                for a in range(len(selected_indices)):
                    if elapsed() > DEADLINE:
                        break
                    i1 = selected_indices[a]
                    for b in range(a+1, len(selected_indices)):
                        if elapsed() > DEADLINE:
                            break
                        i2 = selected_indices[b]
                        mask_no_2 = current_mask & ~(1 << i1) & ~(1 << i2)
                        
                        # Find best pair to add
                        for c in range(len(unselected_indices)):
                            if elapsed() > DEADLINE:
                                break
                            j1 = unselected_indices[c]
                            if not can_add(j1, mask_no_2):
                                continue
                            mask_with_j1 = mask_no_2 | (1 << j1)
                            for d in range(c+1, len(unselected_indices)):
                                j2 = unselected_indices[d]
                                if not can_add(j2, mask_with_j1):
                                    continue
                                gain = weights[j1] + weights[j2] - weights[i1] - weights[i2]
                                if gain > best_2swap_gain:
                                    best_2swap_gain = gain
                                    best_i1, best_i2, best_j1, best_j2 = i1, i2, j1, j2
                        if best_2swap_gain > 1e-12:
                            break
                    if best_2swap_gain > 1e-12:
                        break
                
                if best_i1 >= 0 and best_2swap_gain > 1e-12:
                    new_mask = current_mask & ~(1 << best_i1) & ~(1 << best_i2)
                    new_mask |= (1 << best_j1) | (1 << best_j2)
                    current_mask = new_mask
                    current_sum += best_2swap_gain
                    improved = True
        
        return current_mask, current_sum
    
    # Run local search from best candidate
    selected = best_mask
    selected, sel_sum = local_search(selected, max_iters=150)
    if sel_sum > best_sum:
        best_sum = sel_sum
        best_mask = selected
    
    # Also run local search from the "no 2" greedy result
    if mask_no2 != best_mask:
        sel2, sum2 = local_search(mask_no2, max_iters=80)
        if sum2 > best_sum:
            best_sum = sum2
            best_mask = sel2
    
    # Run local search from mixed greedy
    if mask_mixed != best_mask:
        sel3, sum3 = local_search(mask_mixed, max_iters=80)
        if sum3 > best_sum:
            best_sum = sum3
            best_mask = sel3
    
    # Final: try adding any remaining compatible elements
    final_mask = try_additions(best_mask)
    final_sum = reciprocal_sum_from_mask(final_mask)
    if final_sum > best_sum:
        best_sum = final_sum
        best_mask = final_mask
    
    # Convert to result
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(numbers[i])
    result.sort()
    
    if not result:
        result = [2]
    
    return result
