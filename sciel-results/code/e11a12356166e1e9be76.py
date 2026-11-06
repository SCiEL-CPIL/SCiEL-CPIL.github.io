import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    # Precompute for fast AP checking
    # For each x, precompute all pairs (a,b) such that {a,x,b} forms a 3-AP
    # x is middle: a+b=2x, a!=b
    # x is max: a+2b=x... wait, x=2b-a => a=2b-x
    # x is min: x=2a-b => b=2a-x
    
    def has_3ap_with_x(selected_set, x):
        """Check if adding x to selected_set creates a 3-AP."""
        s = selected_set
        # x is middle: need a,b in s with a+b=2x, a!=b
        two_x = 2 * x
        for a in s:
            b = two_x - a
            if b != a and b in s:
                return True
        # x is max: need a,b in s with x=2b-a, i.e., a=2b-x, a<b
        for b in s:
            a = 2 * b - x
            if a < b and a in s:
                return True
        # x is min: need a,b in s with x=2a-b, i.e., b=2a-x, x<a<b
        for a in s:
            if a > x:
                b = 2 * a - x
                if b in s and b > a:
                    return True
        return False
    
    # Greedy initial: try to add small numbers first (higher reciprocal weight)
    selected = set()
    for x in range(1, 257):
        if not has_3ap_with_x(selected, x):
            selected.add(x)
    
    best_set = set(selected)
    best_sum = sum(Fraction(1, x) for x in best_set)
    
    # Pre-sort candidates by reciprocal weight (descending)
    all_candidates = list(range(1, 257))
    candidates_by_weight = sorted(all_candidates, key=lambda v: -1/v)
    
    # Local search
    while time.time() < deadline:
        improved = False
        
        # Try adding elements (highest weight first)
        for x in candidates_by_weight:
            if time.time() >= deadline:
                break
            if x not in best_set and not has_3ap_with_x(best_set, x):
                best_set.add(x)
                best_sum += Fraction(1, x)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-swap: remove one, add another
        # Remove lowest weight elements first, add highest weight candidates
        best_swap_improved = False
        for remove_x in sorted(best_set, key=lambda v: 1/v):  # remove smallest reciprocal first
            if time.time() >= deadline:
                break
            for add_x in candidates_by_weight:
                if time.time() >= deadline:
                    break
                if add_x in best_set:
                    continue
                # Check if removing remove_x and adding add_x is valid
                temp_set = best_set - {remove_x}
                if not has_3ap_with_x(temp_set, add_x):
                    # Valid swap
                    new_sum = best_sum - Fraction(1, remove_x) + Fraction(1, add_x)
                    if new_sum > best_sum:
                        best_set.remove(remove_x)
                        best_set.add(add_x)
                        best_sum = new_sum
                        improved = True
                        best_swap_improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-swap: remove two, add two
        # More expensive but can escape local optima
        found_2swap = False
        items = list(best_set)
        for i in range(len(items)):
            if time.time() >= deadline:
                break
            for j in range(i+1, len(items)):
                if time.time() >= deadline:
                    break
                rem1, rem2 = items[i], items[j]
                temp_set = best_set - {rem1, rem2}
                for add1 in candidates_by_weight:
                    if time.time() >= deadline:
                        break
                    if add1 in temp_set:
                        continue
                    if not has_3ap_with_x(temp_set, add1):
                        temp_set2 = temp_set | {add1}
                        for add2 in candidates_by_weight:
                            if time.time() >= deadline:
                                break
                            if add2 in temp_set2:
                                continue
                            if not has_3ap_with_x(temp_set2, add2):
                                new_sum = best_sum - Fraction(1, rem1) - Fraction(1, rem2) + Fraction(1, add1) + Fraction(1, add2)
                                if new_sum > best_sum:
                                    best_set = temp_set2 | {add2}
                                    best_sum = new_sum
                                    found_2swap = True
                                    improved = True
                                    break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break
            if found_2swap:
                break
        
        if not improved:
            break
    
    return sorted(best_set)
