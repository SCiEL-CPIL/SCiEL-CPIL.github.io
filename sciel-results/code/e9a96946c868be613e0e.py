import time
import random
from itertools import combinations

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints."""
    n = len(selected)
    
    # If we have fewer than 3 elements, we can't form a sum of 3 distinct others.
    # Also, for the second case, we need 2 distinct others for each a, so n >= 3.
    if n < 3:
        return True

    # Case 1: x is the divisor.
    # Check if there exist distinct i, j, k in selected such that (selected[i] + selected[j] + selected[k]) % x == 0
    for i in range(n):
        a1 = selected[i]
        for j in range(i + 1, n):
            a2 = selected[j]
            s12 = a1 + a2
            for k in range(j + 1, n):
                if (s12 + selected[k]) % x == 0:
                    return False

    # Case 2: x is a summand.
    # For each a in selected, check if there exist distinct b, c in selected\{a}
    # such that (x + b + c) % a == 0.
    for idx_a, a in enumerate(selected):
        # We need pairs (i, j) from selected indices, excluding idx_a
        for i in range(n):
            if i == idx_a:
                continue
            a1 = selected[i]
            for j in range(i + 1, n):
                if j == idx_a:
                    continue
                if (a1 + selected[j] + x) % a == 0:
                    return False

    return True

def _greedy(order, deadline):
    """Greedy construction following given order, respecting deadline."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    start_time = time.time()
    # 60s limit. Use 50s for safety.
    deadline = start_time + 50.0
    
    best = []
    best_size = 0
    
    all_numbers = list(range(1, 193))
    
    # Heuristic 1: Descending (Large numbers are less likely to be divisors of small sums)
    order1 = list(range(192, 0, -1))
    res = _greedy(order1, deadline)
    if len(res) > best_size:
        best = res
        best_size = len(res)
        
    # Heuristic 2: Ascending
    if time.time() < deadline:
        order2 = list(range(1, 193))
        res = _greedy(order2, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)

    # Heuristic 3: Primes first (Primes are hard to divide sums unless the sum is a multiple of the prime)
    if time.time() < deadline:
        primes = [x for x in range(192, 0, -1) if _is_prime(x)]
        non_primes = [x for x in range(192, 0, -1) if not _is_prime(x)]
        order3 = primes + non_primes
        res = _greedy(order3, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)

    # Heuristic 4: Coprime to 6 first
    if time.time() < deadline:
        def coprime6(x):
            return x % 2 != 0 and x % 3 != 0
        c6 = [x for x in range(192, 0, -1) if coprime6(x)]
        rest = [x for x in range(192, 0, -1) if not coprime6(x)]
        order4 = c6 + rest
        res = _greedy(order4, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)

    # Random Restarts
    # Use a random seed for reproducibility if needed, but time-based is fine.
    rng = random.Random(42)
    
    while time.time() < deadline - 1.0: # Leave 1s margin for final return
        # Create a random order
        # Bias: Sometimes prefer larger numbers
        if rng.random() < 0.5:
            order = list(range(192, 0, -1))
        else:
            order = list(range(1, 193))
        rng.shuffle(order)
        
        res = _greedy(order, deadline)
        if len(res) > best_size:
            best = res
            best_size = len(res)
            
            # If we hit the theoretical max or a very high number, we might stop early?
            # Max is 48. If we hit 48, we can return immediately.
            if best_size >= 48:
                return best

    return best
