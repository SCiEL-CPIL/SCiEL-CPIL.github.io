"""Combined program: structural greedy with multiple orderings + efficient incremental checks + time guard."""
import time
import random
from itertools import combinations

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints.
    
    Only new violations can involve x as a summand or as the divisor.
    """
    n = len(selected)
    
    # Case 1: x is the divisor. Need no 3 distinct elements from selected whose sum ≡ 0 mod x.
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        return False
    
    # Case 2: x is a summand. For each a in selected, check if any 2 other elements
    # from selected\{a} plus x have sum ≡ 0 mod a.
    if n >= 2:
        for idx_a, a in enumerate(selected):
            for i in range(n):
                if i == idx_a:
                    continue
                for j in range(i+1, n):
                    if j == idx_a:
                        continue
                    if (selected[i] + selected[j] + x) % a == 0:
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

def _num_divisors(x):
    """Count divisors of x."""
    cnt = 0
    for d in range(1, int(x**0.5) + 1):
        if x % d == 0:
            cnt += 1
            if d != x // d:
                cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # leave 10s safety margin
    
    best = []
    
    # Ordering 1: descending (original)
    order1 = list(range(192, 0, -1))
    result1 = _greedy(order1, deadline)
    best = result1
    
    # Ordering 2: ascending
    if time.time() < deadline:
        order2 = list(range(1, 193))
        result2 = _greedy(order2, deadline)
        if len(result2) > len(best):
            best = result2
    
    # Ordering 3: larger numbers first but skip small ones initially
    if time.time() < deadline:
        order3 = list(range(192, 96, -1)) + list(range(96, 0, -1))
        result3 = _greedy(order3, deadline)
        if len(result3) > len(best):
            best = result3
    
    # Ordering 4: only odd numbers first, then even
    if time.time() < deadline:
        odds = [x for x in range(192, 0, -1) if x % 2 == 1]
        evens = [x for x in range(192, 0, -1) if x % 2 == 0]
        order4 = odds + evens
        result4 = _greedy(order4, deadline)
        if len(result4) > len(best):
            best = result4
    
    # Ordering 5: numbers coprime to 6 first, then rest
    if time.time() < deadline:
        def coprime6(x):
            return x % 2 != 0 and x % 3 != 0
        c6 = [x for x in range(192, 0, -1) if coprime6(x)]
        rest = [x for x in range(192, 0, -1) if not coprime6(x)]
        order5 = c6 + rest
        result5 = _greedy(order5, deadline)
        if len(result5) > len(best):
            best = result5
    
    # Ordering 6: high numbers, prioritize those with few divisors
    if time.time() < deadline:
        order6 = sorted(range(192, 0, -1), key=lambda x: (_num_divisors(x), -x))
        result6 = _greedy(order6, deadline)
        if len(result6) > len(best):
            best = result6
    
    # Ordering 7: Random restarts with different seeds
    if time.time() < deadline:
        rng = random.Random(42)
        for _ in range(3):
            if time.time() > deadline:
                break
            order7 = list(range(1, 193))
            rng.shuffle(order7)
            result7 = _greedy(order7, deadline)
            if len(result7) > len(best):
                best = result7
    
    # Ordering 8: Prime-heavy first
    if time.time() < deadline:
        def is_prime(x):
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
        primes = [x for x in range(192, 0, -1) if is_prime(x)]
        non_primes = [x for x in range(192, 0, -1) if not is_prime(x)]
        order8 = primes + non_primes
        result8 = _greedy(order8, deadline)
        if len(result8) > len(best):
            best = result8
    
    return best
