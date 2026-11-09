"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 88.0

    def time_exceeded():
        return time.time() - start_time > deadline - 2.0

    # Precompute prime factors for all possible sums (2..20000)
    MAX_SUM = 20000
    pf = [set() for _ in range(MAX_SUM + 1)]
    for p in range(2, MAX_SUM + 1):
        if not pf[p]:  # p is prime
            for m in range(p, MAX_SUM + 1, p):
                pf[m].add(p)

    def count_distinct_primes(arr):
        """Count distinct prime factors across all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= pf[arr[i] + arr[j]]
        return len(primes)

    def get_primes_for(arr):
        """Return the set of primes for a given array."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= pf[arr[i] + arr[j]]
        return primes

    # Strategy 1: Powers of 2 (structural) - baseline
    powers_of_2 = [2**i for i in range(12)]
    assert all(x <= 10000 for x in powers_of_2)
    assert len(set(powers_of_2)) == 12

    best_result = list(powers_of_2)
    best_count = count_distinct_primes(best_result)

    # Precompute all 7-smooth numbers up to 20000
    smooth_numbers = set()
    for a in range(15):  # 2^a <= 20000
        v2 = 2**a
        if v2 > MAX_SUM:
            break
        for b in range(10):  # 3^b <= 20000
            v3 = v2 * (3**b)
            if v3 > MAX_SUM:
                break
            for c in range(7):  # 5^c <= 20000
                v5 = v3 * (5**c)
                if v5 > MAX_SUM:
                    break
                for d in range(5):  # 7^d <= 20000
                    v7 = v5 * (7**d)
                    if v7 > MAX_SUM:
                        break
                    if v7 >= 2:
                        smooth_numbers.add(v7)

    # Strategy 2: Find clique where all pairwise sums are 7-smooth
    # Try multiple scaling factors

    def try_scaled_clique(scale, max_a, size=12):
        """Find size distinct a_i in 1..max_a with all pairwise sums 7-smooth, return scaled list or None."""
        if time_exceeded():
            return None
        
        # Build compatibility graph
        compat = {}
        for a in range(1, max_a + 1):
            compat[a] = set()
        for a in range(1, max_a + 1):
            for b in range(a + 1, max_a + 1):
                if (a + b) in smooth_numbers:
                    compat[a].add(b)
                    compat[b].add(a)
        
        # Backtracking with degree-ordered selection
        nodes = list(range(1, max_a + 1))
        
        def backtrack(selected, available):
            if time_exceeded():
                return None
            if len(selected) == size:
                return list(selected)
            
            # Filter available to those compatible with all selected
            filtered = [v for v in available if all(v in compat[s] for s in selected)]
            
            if len(filtered) < size - len(selected):
                return None
            
            # Order by degree descending for better pruning
            filtered.sort(key=lambda v: -len(compat[v]))
            
            for v in filtered:
                new_selected = selected + [v]
                new_available = [u for u in filtered if u > v]
                result = backtrack(new_selected, new_available)
                if result is not None:
                    return result
            return None
        
        a_clique = backtrack([], nodes)
        if a_clique is not None and len(a_clique) == size:
            scaled = [a * scale for a in a_clique]
            if len(set(scaled)) == size and all(1 <= x <= 10000 for x in scaled):
                return scaled
        return None

    # Try scale 256 (max_a=39)
    result = try_scaled_clique(256, 39)
    if result:
        cnt = count_distinct_primes(result)
        if cnt < best_count:
            best_count = cnt
            best_result = list(result)
        if best_count <= 4:
            return best_result

    # Try scale 128 (max_a=78)
    if not time_exceeded():
        result = try_scaled_clique(128, 78)
        if result:
            cnt = count_distinct_primes(result)
            if cnt < best_count:
                best_count = cnt
                best_result = list(result)
            if best_count <= 4:
                return best_result

    # Try scale 64 (max_a=156)
    if not time_exceeded():
        result = try_scaled_clique(64, 156)
        if result:
            cnt = count_distinct_primes(result)
            if cnt < best_count:
                best_count = cnt
                best_result = list(result)
            if best_count <= 4:
                return best_result

    # Strategy 3: Local search to improve current best
    def local_search(arr, max_iter=500):
        """Try swapping elements to reduce prime count."""
        if time_exceeded():
            return list(arr)
        
        current = list(arr)
        current_count = count_distinct_primes(current)
        best = list(current)
        best_cnt = current_count
        
        for _ in range(max_iter):
            if time_exceeded():
                break
            
            # Randomly pick two indices
            i, j = random.sample(range(12), 2)
            
            # Try swapping with a random new value
            new_val = random.randint(1, 10000)
            if new_val in current and current.index(new_val) not in (i, j):
                continue
            
            old_val = current[i]
            current[i] = new_val
            
            new_count = count_distinct_primes(current)
            if new_count < best_cnt:
                best_cnt = new_count
                best = list(current)
            else:
                current[i] = old_val
        
        return best

    if not time_exceeded():
        improved = local_search(best_result, max_iter=200)
        cnt = count_distinct_primes(improved)
        if cnt < best_count:
            best_count = cnt
            best_result = list(improved)

    # Strategy 4: Structured sets (fallbacks)
    fallbacks = [
        list(range(1, 24, 2))[:12],
        [6*(i+1) for i in range(12)],
        list(range(100, 112)),
        list(range(1, 13)),
        [3*(2**i) for i in range(12)],
        [5*(2**i) for i in range(12) if 5*(2**i) <= 10000][:12],
    ]
    
    for fb in fallbacks:
        if len(fb) != 12 or len(set(fb)) != 12 or any(x > 10000 or x < 1 for x in fb):
            continue
        cnt = count_distinct_primes(fb)
        if cnt < best_count:
            best_count = cnt
            best_result = list(fb)

    return best_result
# EVOLVE-BLOCK-END
