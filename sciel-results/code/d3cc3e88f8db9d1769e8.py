"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = 256
    
    # Precompute for each x, the list of pairs (a, c) such that a + c = 2*x, a != c, a != x, c != x
    # and a, c in [1, 256]
    # For can_add: x conflicts with set S if exists a in S such that (2*x - a) in S and 2*x-a != a and 2*x-a != x
    # OR exists a in S such that (2*a - x) in S and 2*a-x != a and 2*a-x != x
    
    # For fast checking, precompute for each x:
    # - left[x] = list of a such that a < x and (2*x - a) in [1,256] and (2*x-a) != a and (2*x-a) != x
    #   meaning if both a and (2*x-a) are in S, adding x creates 3-AP (x as middle)
    # - right[x] = list of a such that a > x and (2*a - x) in [1,256] and (2*a-x) != a and (2*a-x) != x
    #   meaning if both a and (2*a-x) are in S, adding x creates 3-AP (x as endpoint, a as middle)
    
    # Actually, let's think more carefully:
    # Adding x to S creates a 3-AP iff:
    # 1. x is middle: exists a, c in S with a + c = 2x, a != c, a != x, c != x
    # 2. x is left endpoint: exists b, c in S with x + c = 2b, b != x, c != x, b != c => c = 2b - x
    # 3. x is right endpoint: exists a, b in S with a + x = 2b, a != x, b != x, a != b => a = 2b - x
    
    # For case 1: for each a in S, check if (2x - a) in S, with 2x-a != a and 2x-a != x and a != x
    # For case 2: for each b in S, check if (2b - x) in S, with 2b-x != b and 2b-x != x and b != x
    # For case 3: for each b in S, check if (2b - x) in S, with 2b-x != b and 2b-x != x and b != x
    # Wait, cases 2 and 3 are the same check! If x is an endpoint, there's a middle b such that the other endpoint is 2b-x.
    # So really: adding x creates 3-AP iff:
    #   (A) exists a in S: a != x, (2x-a) in S, (2x-a) != a, (2x-a) != x  [x is middle]
    #   (B) exists b in S: b != x, (2b-x) in S, (2b-x) != b, (2b-x) != x  [x is endpoint]
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a 3-AP. O(|S|)"""
        for y in selected_set:
            if y == x:
                continue
            # Check if x is middle: y and (2x-y) both in S
            z = 2 * x - y
            if z != x and z != y and z in selected_set:
                return False
            # Check if y is middle: x and (2y-x) both in S
            z2 = 2 * y - x
            if z2 != x and z2 != y and z2 in selected_set:
                return False
        return True
    
    def is_3ap_free(selected_set):
        """Verify no 3-AP in selected_set. O(K^2) with early exit."""
        s = sorted(selected_set)
        k = len(s)
        for i in range(k):
            a = s[i]
            for j in range(i + 1, k):
                b = s[j]
                c = 2 * b - a
                if c in selected_set and c != a and c != b:
                    return False
        return True
    
    def compute_weight(selected_set):
        total = Fraction(0, 1)
        for x in selected_set:
            total += Fraction(1, x)
        return total
    
    def greedy_construct(order):
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        return selected
    
    # Multiple diverse initializations
    best_set = set()
    best_weight = Fraction(0, 1)
    
    # Strategy 1: Standard increasing greedy
    s1 = greedy_construct(list(range(1, 257)))
    w1 = compute_weight(s1)
    if w1 > best_weight:
        best_set = s1
        best_weight = w1
    
    # Strategy 2: Randomized greedy with bias toward small numbers
    for trial in range(12):
        if time.time() > deadline - 20:
            break
        random.seed(trial * 7919 + 123)
        order = list(range(1, 257))
        # Partial shuffle with bias: smaller numbers more likely to stay early
        for i in range(len(order) - 1):
            if random.random() < 0.5 - 0.3 * (i / len(order)):
                j = i + random.randint(1, len(order) - i - 1)
                order[i], order[j] = order[j], order[i]
        s = greedy_construct(order)
        w = compute_weight(s)
        if w > best_weight:
            best_set = s
            best_weight = w
    
    # Strategy 3: Decreasing then re-add small ones
    s3 = greedy_construct(list(range(256, 0, -1)))
    # Try to add back small numbers
    for x in range(1, 257):
        if x not in s3 and can_add(s3, x):
            s3.add(x)
    w3 = compute_weight(s3)
    if w3 > best_weight:
        best_set = s3
        best_weight = w3
    
    # Local search with multiple phases
    improved = True
    while improved and time.time() < deadline - 8:
        improved = False
        current_set = set(best_set)
        current_weight = best_weight
        
        # Phase 1: Additions - try to add each unselected number
        for x in range(1, 257):
            if x not in current_set and can_add(current_set, x):
                new_set = current_set | {x}
                new_weight = current_weight + Fraction(1, x)
                if new_weight > current_weight:
                    best_set = new_set
                    best_weight = new_weight
                    improved = True
        
        # Phase 2: Single swaps - remove y, add x where 1/x > 1/y
        for y in sorted(best_set, reverse=True):  # Try removing larger elements first
            gain_threshold = Fraction(1, y)
            for x in range(1, 257):
                if x in best_set or x == y:
                    continue
                # Quick check: only worth it if 1/x > 1/y
                if x >= y:
                    continue
                trial_set = (best_set - {y}) | {x}
                if is_3ap_free(trial_set):
                    new_weight = best_weight - Fraction(1, y) + Fraction(1, x)
                    if new_weight > best_weight:
                        best_set = trial_set
                        best_weight = new_weight
                        improved = True
                        break  # Move to next y
        
        # Phase 3: 2-for-1 swaps if time permits
        if time.time() < deadline - 12:
            y_list = sorted(list(best_set), reverse=True)
            for i in range(len(y_list)):
                y1 = y_list[i]
                for j in range(i + 1, len(y_list)):
                    y2 = y_list[j]
                    if time.time() > deadline - 8:
                        break
                    # Only worth it if 1/x > 1/y1 + 1/y2 for some x
                    # Max possible gain: 1/1 = 1, so if 1/y1 + 1/y2 >= 1, skip
                    if Fraction(1, y1) + Fraction(1, y2) >= Fraction(1, 1):
                        continue
                    for x in range(1, 257):
                        if x in best_set or x == y1 or x == y2:
                            continue
                        trial_set = (best_set - {y1, y2}) | {x}
                        if is_3ap_free(trial_set):
                            new_weight = best_weight - Fraction(1, y1) - Fraction(1, y2) + Fraction(1, x)
                            if new_weight > best_weight:
                                best_set = trial_set
                                best_weight = new_weight
                                improved = True
                                break
                    if time.time() > deadline - 8:
                        break
                if time.time() > deadline - 8:
                    break
        
        # Phase 4: 1-for-2 swaps (remove one, add two smaller)
        if time.time() < deadline - 15:
            for y in sorted(best_set, reverse=True):
                if time.time() > deadline - 8:
                    break
                for x1 in range(1, y):
                    if x1 in best_set or x1 == y:
                        continue
                    for x2 in range(x1 + 1, y):
                        if x2 in best_set or x2 == y:
                            continue
                        if time.time() > deadline - 8:
                            break
                        # Quick check: 1/x1 + 1/x2 > 1/y ?
                        if Fraction(1, x1) + Fraction(1, x2) <= Fraction(1, y):
                            continue
                        trial_set = (best_set - {y}) | {x1, x2}
                        if is_3ap_free(trial_set):
                            new_weight = best_weight - Fraction(1, y) + Fraction(1, x1) + Fraction(1, x2)
                            if new_weight > best_weight:
                                best_set = trial_set
                                best_weight = new_weight
                                improved = True
                                break
                    if time.time() > deadline - 8:
                        break
        
        # Phase 5: Try 3-for-1 swaps if time permits
        if time.time() < deadline - 20:
            y_list = sorted(list(best_set), reverse=True)
            for i in range(len(y_list)):
                y1 = y_list[i]
                for j in range(i + 1, len(y_list)):
                    y2 = y_list[j]
                    for k in range(j + 1, len(y_list)):
                        y3 = y_list[k]
                        if time.time() > deadline - 8:
                            break
                        if Fraction(1, y1) + Fraction(1, y2) + Fraction(1, y3) >= Fraction(1, 1):
                            continue
                        for x in range(1, 257):
                            if x in best_set or x == y1 or x == y2 or x == y3:
                                continue
                            trial_set = (best_set - {y1, y2, y3}) | {x}
                            if is_3ap_free(trial_set):
                                new_weight = best_weight - Fraction(1, y1) - Fraction(1, y2) - Fraction(1, y3) + Fraction(1, x)
                                if new_weight > best_weight:
                                    best_set = trial_set
                                    best_weight = new_weight
                                    improved = True
                                    break
                        if time.time() > deadline - 8:
                            break
                    if time.time() > deadline - 8:
                        break
                if time.time() > deadline - 8:
                    break
    
    # Final polish: try all single additions one more time
    for x in range(1, 257):
        if x not in best_set and can_add(best_set, x):
            best_set = best_set | {x}
            best_weight += Fraction(1, x)
    
    # Verify final result is 3-AP-free
    if not is_3ap_free(best_set):
        # Fallback to a known valid construction
        fallback = set()
        for x in range(1, 257):
            if can_add(fallback, x):
                fallback.add(x)
        best_set = fallback
    
    result = sorted(best_set)
    if not result:
        return [1]
    return result
# EVOLVE-BLOCK-END
