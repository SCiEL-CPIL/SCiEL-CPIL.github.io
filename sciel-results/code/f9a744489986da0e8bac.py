import time
import random
import math

def _objective(A):
    """
    Compute max(|A+A|, |A*A|) using set operations.
    """
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _generate_5_smooth(limit):
    """Generate all 5-smooth numbers up to limit."""
    smooths = set([1])
    powers2 = [1]
    powers3 = [1]
    powers5 = [1]
    
    # Generate powers
    while powers2[-1] * 2 <= limit:
        powers2.append(powers2[-1] * 2)
    while powers3[-1] * 3 <= limit:
        powers3.append(powers3[-1] * 3)
    while powers5[-1] * 5 <= limit:
        powers5.append(powers5[-1] * 5)
        
    for p2 in powers2:
        for p3 in powers3:
            val = p2 * p3
            if val > limit:
                break
            for p5 in powers5:
                prod = val * p5
                if prod > limit:
                    break
                smooths.add(prod)
                
    return sorted(smooths)

def _local_search(A, smooths, deadline):
    """
    Local search restricting moves to 5-smooth numbers.
    """
    current = list(A)
    current_cost = _objective(current)
    best = list(current)
    best_cost = current_cost
    
    smooth_set = set(smooths)
    # Pre-filter smooths to those <= 10000
    valid_smooths = [x for x in smooths if x <= 10000]
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        indices = list(range(20))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline - 2:
                break
            old = current[idx]
            # Try replacing with other 5-smooth numbers
            # Sort valid_smooths by distance to old to find local improvements faster?
            # Or just iterate.
            for new_val in valid_smooths:
                if new_val == old:
                    continue
                if new_val in current:
                    continue
                current[idx] = new_val
                new_cost = _objective(current)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = list(current)
                    break
                else:
                    current[idx] = old
            if time.time() > deadline - 2:
                break
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    random.seed(42)
    
    # Generate 5-smooth numbers
    smooths = _generate_5_smooth(10000)
    
    best_set = None
    best_cost = float('inf')
    
    # Initial candidates: Windows of 20 consecutive 5-smooth numbers
    for i in range(len(smooths) - 20 + 1):
        window = smooths[i:i+20]
        if not _is_valid(window):
            continue
        cost = _objective(window)
        if cost < best_cost:
            best_cost = cost
            best_set = list(window)
            
    if best_set is None:
        # Fallback
        best_set = list(range(1, 21))
        best_cost = _objective(best_set)
        
    # Local Search Phase
    # Run multiple local searches with different starting points
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            start_A = list(best_set)
        else:
            # Perturb best set by swapping a few elements with other smooths
            start_A = list(best_set)
            for _ in range(random.randint(1, 5)):
                idx = random.randrange(20)
                # Pick a random smooth
                new_val = random.choice(smooths)
                if new_val not in start_A:
                    start_A[idx] = new_val
            start_A = sorted(start_A)
            if not _is_valid(start_A):
                continue
                
        result, cost = _local_search(start_A, smooths, deadline)
        if cost < best_cost:
            best_cost = cost
            best_set = result
            
    return best_set
