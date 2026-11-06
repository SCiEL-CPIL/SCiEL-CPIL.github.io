"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

DEADLINE = 85.0  # safety margin before 90s kill

def _gcd_matrix(limit):
    """Precompute gcd for all pairs up to limit."""
    m = [[0] * (limit + 1) for _ in range(limit + 1)]
    for i in range(1, limit + 1):
        for j in range(i, limit + 1):
            m[i][j] = gcd(i, j)
            m[j][i] = m[i][j]
    return m

def _can_add(selected, x, gcd_m):
    """Check if adding x to selected keeps the set valid. Uses direct table lookups."""
    n = len(selected)
    for i in range(n):
        a = selected[i]
        g_ax = gcd_m[a][x]
        for j in range(i + 1, n):
            b = selected[j]
            g_ab = gcd_m[a][b]
            g_bx = gcd_m[b][x]
            if g_ab == g_ax == g_bx:
                return False
    return True

def _is_valid_set(selected, gcd_m):
    """Check if selected set has no three distinct elements with all pairwise gcds equal."""
    n = len(selected)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                g1 = gcd_m[selected[i]][selected[j]]
                g2 = gcd_m[selected[i]][selected[k]]
                g3 = gcd_m[selected[j]][selected[k]]
                if g1 == g2 == g3:
                    return False
    return True

def _greedy_construct(gcd_m, limit, start_order=None):
    """Greedy construction: add elements in given order if valid."""
    if start_order is None:
        start_order = list(range(1, limit + 1))
    selected = []
    for x in start_order:
        if _can_add(selected, x, gcd_m):
            selected.append(x)
    return selected

def _local_search(selected, gcd_m, limit, time_budget):
    """Local search with multiple strategies: add, swap, 2-swap, and simulated annealing moves."""
    start = time.time()
    best = list(selected)
    best_size = len(best)
    random.seed(42)  # deterministic for reproducibility
    
    while time.time() - start < time_budget:
        improved = False
        
        # Strategy 1: Try adding each unused element
        used = set(best)
        for x in range(1, limit + 1):
            if x not in used and _can_add(best, x, gcd_m):
                best.append(x)
                best_size += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Strategy 2: Single swap - remove one, try to add two
        n_best = len(best)
        found_improvement = False
        for i in range(n_best):
            if time.time() - start > time_budget:
                break
            removed = best[i]
            base = best[:i] + best[i+1:]
            
            # Try each candidate replacement
            for x in range(1, limit + 1):
                if x == removed or x in base:
                    continue
                if not _can_add(base, x, gcd_m):
                    continue
                
                swapped = base + [x]
                
                # Try to add any additional element
                for y in range(1, limit + 1):
                    if y in swapped:
                        continue
                    if _can_add(swapped, y, gcd_m):
                        # Accept this swap since it enables growth
                        best = swapped + [y]
                        best_size += 1
                        improved = True
                        found_improvement = True
                        break
                if found_improvement:
                    break
            if found_improvement:
                break
        
        if improved:
            continue
        
        # Strategy 3: Double swap - remove two, try to add three
        if time.time() - start < time_budget:
            found_improvement = False
            for i in range(len(best)):
                if time.time() - start > time_budget:
                    break
                for j in range(i + 1, len(best)):
                    if time.time() - start > time_budget:
                        break
                    removed1 = best[i]
                    removed2 = best[j]
                    base = [x for idx, x in enumerate(best) if idx != i and idx != j]
                    
                    # Try pairs of replacements
                    for x in range(1, limit + 1):
                        if x in base or x == removed1 or x == removed2:
                            continue
                        if not _can_add(base, x, gcd_m):
                            continue
                        base_x = base + [x]
                        
                        for y in range(1, limit + 1):
                            if y in base_x or y == removed1 or y == removed2:
                                continue
                            if not _can_add(base_x, y, gcd_m):
                                continue
                            swapped = base_x + [y]
                            
                            # Try to add one more
                            for z in range(1, limit + 1):
                                if z in swapped or z == removed1 or z == removed2:
                                    continue
                                if _can_add(swapped, z, gcd_m):
                                    best = swapped + [z]
                                    best_size += 1
                                    improved = True
                                    found_improvement = True
                                    break
                            if found_improvement:
                                break
                        if found_improvement:
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            
            if improved:
                continue
        
        # Strategy 4: Simulated annealing-like move - accept same-size swaps with probability
        if time.time() - start < time_budget:
            # Try random swaps that maintain size but might enable future growth
            for _ in range(10):
                if time.time() - start > time_budget:
                    break
                i = random.randint(0, len(best) - 1)
                x = random.randint(1, limit)
                if x in best:
                    continue
                removed = best[i]
                base = best[:i] + best[i+1:]
                if _can_add(base, x, gcd_m):
                    # Accept with some probability to explore
                    if random.random() < 0.3:
                        best = base + [x]
        
        if not improved:
            # Try a few more random perturbations before giving up
            for _ in range(5):
                if time.time() - start > time_budget:
                    break
                # Randomly remove 1-2 elements and try to rebuild
                num_remove = random.randint(1, 2)
                indices = random.sample(range(len(best)), min(num_remove, len(best)))
                indices.sort(reverse=True)
                for idx in indices:
                    del best[idx]
                best_size = len(best)
                
                # Try to add back
                used = set(best)
                for x in range(1, limit + 1):
                    if x not in used and _can_add(best, x, gcd_m):
                        best.append(x)
                        best_size += 1
                        break
                
                if best_size > len(best):
                    break
            
            if not improved:
                break
    
    return best

def _divisor_count(n):
    """Count number of divisors of n."""
    count = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def _largest_prime_factor(n):
    """Find the largest prime factor of n."""
    if n < 2:
        return 1
    largest = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            largest = d
            n //= d
        d += 1
    if n > 1:
        largest = n
    return largest

def _radical(n):
    """Compute the radical of n (product of distinct prime factors)."""
    if n < 2:
        return 1
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            result *= d
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        result *= n
    return result

def construct():
    start_time = time.time()
    limit = 128
    gcd_m = _gcd_matrix(limit)
    
    # Precompute number-theoretic properties for ordering
    is_prime = [False] * (limit + 1)
    for x in range(2, limit + 1):
        is_prime[x] = all(x % d for d in range(2, int(x**0.5) + 1))
    
    # Generate diverse orderings
    orderings = []
    
    # Basic orderings
    orderings.append(list(range(1, limit + 1)))
    orderings.append(list(range(limit, 0, -1)))
    
    # Primes first, then composites
    primes = [x for x in range(2, limit + 1) if is_prime[x]]
    composites = [x for x in range(1, limit + 1) if x != 1 and not is_prime[x]]
    orderings.append([1] + primes + composites)
    
    # By number of divisors (fewer divisors first - less likely to create equal gcds)
    by_divcount = sorted(range(1, limit + 1), key=lambda x: _divisor_count(x))
    orderings.append(by_divcount)
    
    # By largest prime factor (larger first)
    by_lpf = sorted(range(1, limit + 1), key=lambda x: -_largest_prime_factor(x))
    orderings.append(by_lpf)
    
    # By radical (smaller first)
    by_rad = sorted(range(1, limit + 1), key=lambda x: _radical(x))
    orderings.append(by_rad)
    
    # Odd first, then even
    orderings.append([x for x in range(1, limit + 1, 2)] + [x for x in range(2, limit + 1, 2)])
    
    # Even first, then odd
    orderings.append([x for x in range(2, limit + 1, 2)] + [x for x in range(1, limit + 1, 2)])
    
    # Powers of 2 first
    powers_of_2 = [2**i for i in range(1, 8) if 2**i <= limit]
    rest = [x for x in range(1, limit + 1) if x not in set(powers_of_2)]
    orderings.append(powers_of_2 + rest)
    
    # Multiples of small primes first
    for p in [2, 3, 5, 7]:
        multiples = [x for x in range(1, limit + 1) if x % p == 0]
        non_multiples = [x for x in range(1, limit + 1) if x % p != 0]
        orderings.append(multiples + non_multiples)
    
    # Random orderings (deterministic with fixed seed)
    random.seed(123)
    for _ in range(5):
        rand_order = list(range(1, limit + 1))
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    best = []
    best_size = 0
    
    # Run greedy with each ordering, keep the best
    for order in orderings:
        if time.time() - start_time > DEADLINE - 20:
            break
        result = _greedy_construct(gcd_m, limit, order)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Local search on the best result
    remaining = DEADLINE - (time.time() - start_time) - 2
    if remaining > 1:
        improved = _local_search(best, gcd_m, limit, remaining)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # If we have time, try local search on second-best greedy result too
    if time.time() - start_time < DEADLINE - 10 and best_size < limit:
        # Find second best greedy result
        second_best = []
        second_best_size = 0
        for order in orderings:
            if time.time() - start_time > DEADLINE - 10:
                break
            if order == orderings[0] and best == _greedy_construct(gcd_m, limit, order):
                continue  # skip the one we already used
            result = _greedy_construct(gcd_m, limit, order)
            if len(result) > second_best_size and result != best:
                second_best = result
                second_best_size = len(result)
        
        if second_best_size > 0:
            remaining2 = DEADLINE - (time.time() - start_time) - 2
            if remaining2 > 1:
                improved2 = _local_search(second_best, gcd_m, limit, remaining2)
                if len(improved2) > best_size:
                    best = improved2
                    best_size = len(improved2)
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
