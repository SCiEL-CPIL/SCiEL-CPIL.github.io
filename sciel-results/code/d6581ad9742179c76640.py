import time
import random
import math

def _get_subset_sums(a):
    """Generate all subset sums for list a."""
    n = len(a)
    sums = [0]
    for i in range(n):
        ai = a[i]
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

def _is_valid(a):
    """Check if all subset sums are distinct and elements are distinct positive integers."""
    n = len(a)
    if len(set(a)) != n:
        return False
    if any(x <= 0 for x in a):
        return False
    sums = _get_subset_sums(a)
    return len(set(sums)) == (1 << n)

def _greedy_construct(n):
    """Greedy construction: pick smallest possible value for each position."""
    a = [1]
    for i in range(1, n):
        # Try values starting from 1
        for val in range(1, 1 << 20):
            candidate = a + [val]
            if len(set(candidate)) != len(candidate):
                continue
            if _is_valid(candidate):
                a = candidate
                break
        else:
            # Should not happen for reasonable n
            a = a + [1 << (i + 1)]
    return a

def construct():
    n = 12
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Try greedy construction first
    try:
        a_greedy = _greedy_construct(n)
        cost_greedy = _cost(a_greedy)
    except Exception:
        a_greedy = [1 << i for i in range(n)]
        cost_greedy = _cost(a_greedy)
    
    # Start with powers of 2 as well
    a_pow2 = [1 << i for i in range(n)]
    cost_pow2 = _cost(a_pow2)
    
    # Use the better starting point
    if cost_greedy < cost_pow2:
        current_a = a_greedy[:]
        current_cost = cost_greedy
    else:
        current_a = a_pow2[:]
        current_cost = cost_pow2
    
    best_a = current_a[:]
    best_cost = current_cost
    
    # Simulated Annealing with better parameters
    T = 100.0
    cooling_rate = 0.9995
    min_T = 0.5
    
    # Also try some deterministic improvements: try to reduce the largest element
    # by checking if we can replace it with a smaller value
    
    while time.time() < deadline and T > min_T:
        # Generate neighbor: pick a random position and try a new value
        i = random.randint(0, n - 1)
        
        # Bias towards reducing large values
        if i == n - 1 and current_a[i] > 100:
            # Try to reduce the largest element more aggressively
            delta = random.randint(-max(1, current_a[i] // 10), max(1, current_a[i] // 5))
        else:
            delta = random.randint(-20, 20)
        
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
        
    # Final polish: try to reduce the maximum element by checking nearby values
    if time.time() < deadline:
        max_idx = best_a.index(max(best_a))
        max_val = best_a[max_idx]
        
        # Try reducing the max element
        for trial_val in range(max_val - 1, 0, -1):
            if time.time() > deadline - 5.0:
                break
            candidate = best_a[:]
            candidate[max_idx] = trial_val
            if len(set(candidate)) == n and all(x > 0 for x in candidate):
                if _is_valid(candidate):
                    best_a = candidate
                    best_cost = _cost(candidate)
                    break
    
    if best_cost >= 10000:
        # Fallback to powers of 2
        return [1 << i for i in range(n)]
        
    return best_a
