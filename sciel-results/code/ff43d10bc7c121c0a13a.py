import time
import random
import itertools
from fractions import Fraction

def _is_dissociated(subset):
    """Check if all reciprocal subset sums are distinct."""
    n = len(subset)
    if n == 0:
        return True
    # Use Fraction for exact arithmetic
    # Subset sums of reciprocals: sum of 1/x for x in chosen subset
    # We need all 2^n sums to be distinct
    # For n >= 15, 2^n > 32768, checking all pairs is expensive
    # Use a set of Fractions
    sums = set()
    # Start with empty sum = 0
    sums.add(Fraction(0, 1))
    # Build up: for each element, add it to all existing sums
    # Actually, let's use the standard subset sum enumeration
    # For efficiency, use the incremental approach
    current_sums = [Fraction(0, 1)]
    for x in subset:
        recip = Fraction(1, x)
        new_sums = []
        for s in current_sums:
            ns = s + recip
            if ns in sums:
                return False
            sums.add(ns)
            new_sums.append(ns)
        current_sums.extend(new_sums)
    return True

def _check_fast(subset):
    """Faster check using integer arithmetic.
    If we multiply all reciprocal subset sums by LCM of denominators,
    we get integer subset sums. Two reciprocal subset sums are equal
    iff their scaled versions are equal.
    """
    n = len(subset)
    if n == 0:
        return True
    # LCM of all elements in subset
    from math import gcd
    lcm_val = 1
    for x in subset:
        lcm_val = lcm_val * x // gcd(lcm_val, x)
    # Integer weights: w_i = lcm_val / x_i
    weights = [lcm_val // x for x in subset]
    # Check if all subset sums of weights are distinct
    # Use the incremental set approach
    sums = set()
    sums.add(0)
    current = [0]
    for w in weights:
        new_sums = []
        for s in current:
            ns = s + w
            if ns in sums:
                return False
            sums.add(ns)
            new_sums.append(ns)
        current.extend(new_sums)
    return True

def _check_bitset(subset):
    """Check using bitset approach for moderate n."""
    n = len(subset)
    if n == 0:
        return True
    from math import gcd
    lcm_val = 1
    for x in subset:
        lcm_val = lcm_val * x // gcd(lcm_val, x)
    weights = [lcm_val // x for x in subset]
    # Use a set of integers
    seen = {0}
    current = [0]
    for w in weights:
        new_vals = []
        for s in current:
            ns = s + w
            if ns in seen:
                return False
            seen.add(ns)
            new_vals.append(ns)
        current.extend(new_vals)
    return True

def construct():
    start_time = time.time()
    deadline = 75.0  # safety margin before 90s kill
    
    # Start with the known 9-element solution
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best)
    
    # Try to find larger solutions
    # Strategy: greedy construction + local search
    
    # First, try greedy: add elements one by one
    def greedy_construct(start_set, candidates, deadline):
        """Greedily add elements to start_set."""
        current = list(start_set)
        for cand in candidates:
            if time.time() > deadline:
                break
            if cand in current:
                continue
            trial = current + [cand]
            if _check_bitset(trial):
                current = trial
        return current
    
    # Try different orderings and starting points
    all_elements = list(range(1, 23))
    
    # Try greedy from empty
    for _ in range(50):
        if time.time() > deadline:
            break
        random.shuffle(all_elements)
        result = greedy_construct([], all_elements, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Try local search: start from best, try to add/remove/swap
    if time.time() > deadline:
        return best
    
    def local_search(start_set, deadline):
        """Local search: try adding, removing, or swapping elements."""
        current = list(start_set)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding
            for cand in all_elements:
                if time.time() > deadline:
                    return current
                if cand in current_set:
                    continue
                trial = current + [cand]
                if _check_bitset(trial):
                    current = trial
                    current_set.add(cand)
                    improved = True
                    break
            # Try swapping: remove one, add another
            if not improved:
                for remove_idx in range(len(current)):
                    if time.time() > deadline:
                        return current
                    for cand in all_elements:
                        if cand in current_set:
                            continue
                        trial = [x for i, x in enumerate(current) if i != remove_idx] + [cand]
                        if _check_bitset(trial):
                            current = trial
                            current_set = set(current)
                            improved = True
                            break
                    if improved:
                        break
        return current
    
    # Run local search from best
    result = local_search(best, deadline)
    if len(result) > best_size:
        best = result
        best_size = len(best)
    
    # Try more aggressive search: random restarts with local search
    while time.time() < deadline - 5:
        random_start = random.sample(all_elements, min(8, len(all_elements)))
        result = local_search(random_start, deadline - 5)
        if len(result) > best_size:
            best = result
            best_size = len(best)
    
    return best
