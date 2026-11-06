import time
import random
import math

def _get_subset_sums(a):
    """Generate all subset sums for list a."""
    n = len(a)
    sums = [0]
    for i in range(n):
        ai = a[i]
        # Use list comprehension to add ai to existing sums
        # sums = sums + [s + ai for s in sums]
        # This is O(2^n) per step, total O(n 2^n)
        new_sums = [s + ai for s in sums]
        sums.extend(new_sums)
    return sums

def _cost(a):
    """Calculate cost: max element + penalty for collisions."""
    n = len(a)
    sums = _get_subset_sums(a)
    unique_sums = len(set(sums))
    collisions = (1 << n) - unique_sums
    max_val = max(a)
    # Large penalty for collisions
    return max_val + 10000 * collisions

def construct():
    n = 12
    # Start with powers of 2
    a = [1 << i for i in range(n)]
    best_a = a[:]
    best_cost = _cost(a)
    
    # Simulated Annealing
    T = 50.0
    cooling_rate = 0.999
    min_T = 0.1
    deadline = time.time() + 80.0
    
    current_a = a[:]
    current_cost = best_cost
    
    while time.time() < deadline and T > min_T:
        # Generate neighbor
        i = random.randint(0, n - 1)
        # Random delta, biased towards small changes
        delta = random.randint(-10, 10)
        new_val = current_a[i] + delta
        
        if new_val <= 0:
            T *= cooling_rate
            continue
            
        new_a = current_a[:]
        new_a[i] = new_val
        
        # Check distinctness of elements
        if len(set(new_a)) != n:
            T *= cooling_rate
            continue
            
        new_cost = _cost(new_a)
        
        # Acceptance probability
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_a = new_a
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_a = current_a[:]
                
        T *= cooling_rate
        
    # Ensure we return a valid solution
    # If best_cost has collisions (best_cost >= 10000), we failed to find a valid one
    # But powers of 2 is always valid, so best_a should be valid if we started there and only updated on improvement
    # However, if we got stuck in a local minimum with collisions?
    # No, we started with valid (cost 2048).
    # Any valid solution has cost < 10000.
    # If best_cost >= 10000, it means we found a solution with collisions that had lower "max_val + penalty"?
    # No, penalty is 10000 * collisions.
    # Valid cost: max_val.
    # Invalid cost: max_val + 10000 * k.
    # Since max_val is small (< 10000), any valid solution has cost < 10000.
    # Any invalid solution has cost >= 10000.
    # So if best_cost < 10000, best_a is valid.
    
    if best_cost >= 10000:
        # Fallback to powers of 2
        return [1 << i for i in range(n)]
        
    return best_a
