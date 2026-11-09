import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(numbers):
    """Precompute LCM table for all pairs in [1, 128]."""
    n = max(numbers)
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _greedy_construct_fast(order, lcm_table):
    """
    Greedily add numbers in the given order if they don't form a bad triple.
    Uses precomputed LCM table for speed.
    """
    selected = []
    for x in order:
        ok = True
        n_sel = len(selected)
        for i in range(n_sel):
            a = selected[i]
            la = lcm_table[a][x]
            for j in range(i + 1, n_sel):
                b = selected[j]
                if la == lcm_table[a][b] == lcm_table[b][x]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)
    return selected

def _local_search_improve(subset, lcm_table, deadline):
    """
    Try to improve the subset by swapping elements or adding new ones.
    """
    selected = list(subset)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        # Try adding each excluded number
        excluded = [x for x in range(1, 129) if x not in selected]
        for x in excluded:
            if time.time() > deadline:
                break
            ok = True
            n_sel = len(selected)
            for i in range(n_sel):
                a = selected[i]
                la = lcm_table[a][x]
                for j in range(i + 1, n_sel):
                    b = selected[j]
                    if la == lcm_table[a][b] == lcm_table[b][x]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.append(x)
                improved = True
                break
        # Try swapping: remove one element, try to add a different one
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            candidate_remove = selected[i]
            remaining = [selected[j] for j in range(len(selected)) if j != i]
            for x in range(1, 129):
                if x == candidate_remove or x in remaining:
                    continue
                ok = True
                n_rem = len(remaining)
                for a_idx in range(n_rem):
                    a = remaining[a_idx]
                    la = lcm_table[a][x]
                    for b_idx in range(a_idx + 1, n_rem):
                        b = remaining[b_idx]
                        if la == lcm_table[a][b] == lcm_table[b][x]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    new_set = remaining + [x]
                    if len(new_set) > len(selected):
                        selected = new_set
                        improved = True
                        break
            if improved:
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    numbers = list(range(1, 129))
    
    # Precompute LCM table once
    lcm_table = _precompute_lcms(numbers)
    
    # Strategy 1: Primes and Prime Powers first
    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    # Order 1: Primes, Prime Powers, Others (sorted by value)
    order1 = primes + prime_powers + others
    result1 = _greedy_construct_fast(order1, lcm_table)
    
    # Order 2: Standard 1..128
    order2 = numbers
    result2 = _greedy_construct_fast(order2, lcm_table)
    
    # Order 3: Sorted by number of divisors (ascending)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    order3 = sorted(numbers, key=num_divisors)
    result3 = _greedy_construct_fast(order3, lcm_table)
    
    best_result = max([result1, result2, result3], key=len)
    
    # Try many random shuffles if time permits
    random.seed(42)
    for i in range(500):
        if time.time() > deadline - 10:  # Leave 10s for local search
            break
        random.shuffle(numbers)
        res = _greedy_construct_fast(numbers, lcm_table)
        if len(res) > len(best_result):
            best_result = res
    
    # Local search improvement on the best result
    if time.time() < deadline - 2:
        best_result = _local_search_improve(best_result, lcm_table, deadline - 2)
    
    return best_result
