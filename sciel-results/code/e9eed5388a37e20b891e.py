import math
from itertools import combinations, product

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Systematic search over base-B expansion sets
    # We'll explore different bases and digit patterns
    
    for B in range(3, 25):
        for n in range(2, 12):
            # Generate all possible base-B binary numbers with n digits
            # Each digit is 0 or 1
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            # Try the full set
            U_full = all_vals
            s_full = calc_score(U_full)
            if s_full > best_score:
                best_score = s_full
                best = U_full
            
            # For smaller sets, try various subset patterns
            # Focus on patterns that maximize difference set growth
            if n <= 8:
                # Try greedy addition of elements
                # Start with {0} and add elements that maximize score improvement
                current_set = {0}
                remaining = set(all_vals) - {0}
                
                for _ in range(min(8, len(remaining))):
                    best_elem = None
                    best_s = best_score
                    for elem in list(remaining):
                        test_set = current_set | {elem}
                        s_test = calc_score(list(test_set))
                        if s_test > best_s:
                            best_s = s_test
                            best_elem = elem
                    if best_elem is None:
                        break
                    current_set.add(best_elem)
                    remaining.discard(best_elem)
                    
                    if best_s > best_score:
                        best_score = best_s
                        best = list(current_set)
                
                # Also try random subsets
                import random
                random.seed(42)
                for trial in range(20):
                    subset_size = random.randint(3, min(10, len(all_vals)))
                    subset = random.sample(all_vals, subset_size)
                    if 0 not in subset:
                        subset.append(0)
                    s_sub = calc_score(subset)
                    if s_sub > best_score:
                        best_score = s_sub
                        best = subset

    return best
