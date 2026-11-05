import time
import random
import math
from itertools import combinations

def _factorize(n):
    """Return a set of prime factors of n."""
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            factors.add(f)
            while n % f == 0:
                n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(nums):
    """Count distinct prime factors of all pairwise sums."""
    prime_set = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            prime_set.update(_factorize(s))
    return len(prime_set), prime_set

def _is_smooth(n, primes):
    """Check if n is composed only of primes in the given set."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def _generate_candidates_for_primes(prime_set, limit=10000):
    """
    Generate numbers <= limit such that for any two distinct numbers a, b,
    a+b is composed only of primes in prime_set.
    We do this by generating all S-smooth numbers up to 2*limit,
    then checking which pairs have S-smooth sums.
    """
    # Generate all S-smooth numbers up to 2*limit
    smooth_numbers = []
    
    def gen_smooth(current, idx, max_val):
        if current > max_val:
            return
        if idx == len(prime_set):
            if 1 < current <= max_val:
                smooth_numbers.append(current)
            return
        p = prime_set[idx]
        power = 1
        while power * p <= max_val:
            power *= p
            gen_smooth(current * power, idx + 1, max_val)
        gen_smooth(current, idx + 1, max_val)
    
    gen_smooth(1, 0, 2 * limit)
    
    # Filter to numbers <= limit
    smooth_numbers = [x for x in smooth_numbers if x <= limit]
    
    # Now find which pairs have S-smooth sums
    # A number a is "good" if for many other smooth numbers b, a+b is S-smooth
    # But we want to find a set of 12 numbers where ALL pairwise sums are S-smooth
    
    # Better approach: generate candidate numbers, then use backtracking
    # to find 12 numbers where all pairwise sums are S-smooth
    
    # First, let's identify numbers that can participate in S-smooth sums
    # For a number a to be in our set, there must exist 11 other numbers b1,...,b11
    # such that a+bi is S-smooth for all i
    
    # Let's just return all smooth numbers and let backtracking handle the rest
    return sorted(set(smooth_numbers))

def _find_12_smooth_sumset(prime_set, limit=10000, deadline=None):
    """
    Find 12 distinct positive integers <= limit such that all pairwise sums
    are composed only of primes in prime_set.
    """
    # Generate all S-smooth numbers up to 2*limit
    smooth_sums = set()
    
    def gen_smooth_sums(current, idx, max_val):
        if current > max_val:
            return
        if idx == len(prime_set):
            if current >= 2:  # minimum sum of two positive integers
                smooth_sums.add(current)
            return
        p = prime_set[idx]
        power = 1
        while power * p <= max_val:
            power *= p
            gen_smooth_sums(current * power, idx + 1, max_val)
        gen_smooth_sums(current, idx + 1, max_val)
    
    gen_smooth_sums(1, 0, 2 * limit)
    
    # Now, for each candidate number a (1 to limit), check if it can be part of a set
    # where all pairwise sums are in smooth_sums
    
    # Build adjacency: a and b are compatible if a+b is in smooth_sums
    # We need to find a clique of size 12
    
    # To make this efficient, precompute for each a, the set of compatible b's
    # But that's O(limit^2) which is too slow for limit=10000
    
    # Better: only consider numbers that are "likely" to work
    # A number a is promising if there are many b's such that a+b is S-smooth
    
    # Let's generate candidates more cleverly
    # For each a, the compatible b's are those where a+b is S-smooth
    # So b must be in {s - a : s in smooth_sums, 1 <= s-a <= limit, s-a != a}
    
    # Let's build a list of candidates and their compatibility counts
    candidates = []
    for a in range(1, limit + 1):
        if deadline and time.time() > deadline:
            break
        # Count how many b's are compatible with a
        count = 0
        for s in smooth_sums:
            b = s - a
            if 1 <= b <= limit and b != a:
                count += 1
        if count >= 11:  # Need at least 11 other numbers
            candidates.append((a, count))
    
    # Sort by count descending
    candidates.sort(key=lambda x: -x[1])
    
    # Take top candidates for backtracking
    top_candidates = [c[0] for c in candidates[:200]]
    
    # Backtracking to find clique of size 12
    def backtrack(start_idx, selected, remaining):
        if deadline and time.time() > deadline:
            return None
        if len(selected) == 12:
            return selected.copy()
        
        if start_idx >= len(top_candidates):
            return None
        
        # Try to add top_candidates[start_idx]
        a = top_candidates[start_idx]
        
        # Check if a is compatible with all currently selected
        compatible = True
        for b in selected:
            if (a + b) not in smooth_sums:
                compatible = False
                break
        
        if compatible:
            new_selected = selected + [a]
            result = backtrack(start_idx + 1, new_selected, remaining - 1)
            if result:
                return result
        
        # Try skipping a
        return backtrack(start_idx + 1, selected, remaining)
    
    result = backtrack(0, [], 12)
    if result:
        return sorted(result)
    
    return None

def _local_search_improve(initial_set, prime_set, limit=10000, deadline=None):
    """
    Local search to improve a set by replacing elements.
    """
    current = sorted(initial_set)
    current_count, current_primes = _count_distinct_primes(current)
    
    best = current.copy()
    best_count = current_count
    
    # Generate candidate replacements
    # Focus on numbers that are S-smooth or close to it
    candidates = list(range(1, limit + 1))
    
    while deadline and time.time() < deadline:
        # Pick random index to replace
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        # Try random candidate
        new_val = random.choice(candidates)
        
        if new_val == old_val or new_val in current:
            continue
        
        new_set = current.copy()
        new_set[idx] = new_val
        new_set.sort()
        
        new_count, new_primes = _count_distinct_primes(new_set)
        
        if new_count < best_count:
            best_count = new_count
            best = new_set
            current = new_set
            current_count = new_count
            current_primes = new_primes
        elif new_count == best_count and random.random() < 0.1:
            current = new_set
            current_count = new_count
            current_primes = new_primes
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin (90s limit)
    
    # Strategy: Try different small prime sets and find 12 numbers where
    # all pairwise sums are S-smooth for that set S
    
    # Start with very small prime sets
    prime_sets_to_try = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
    ]
    
    best_result = None
    best_count = float('inf')
    
    for prime_set in prime_sets_to_try:
        if time.time() > deadline:
            break
        
        # Try to find 12 numbers where all pairwise sums are S-smooth
        result = _find_12_smooth_sumset(prime_set, 10000, deadline)
        
        if result:
            count, primes = _count_distinct_primes(result)
            if count < best_count:
                best_count = count
                best_result = result
                # If we found a set with very few primes, great!
                if count <= len(prime_set):
                    break  # Can't do better than this for this prime set
    
    # If we didn't find a perfect S-smooth set, fall back to local search
    if best_result is None or time.time() < deadline:
        # Generate initial set from smooth numbers
        candidates = []
        for i in range(15):
            p2 = 2 ** i
            for j in range(15):
                p3 = 3 ** j
                prod = p2 * p3
                if prod > 10000:
                    break
                candidates.append(prod)
        
        candidates = sorted(set(candidates))
        
        if len(candidates) >= 12:
            initial = random.sample(candidates, 12)
            improved, count = _local_search_improve(initial, [2, 3, 5], 10000, deadline)
            
            if best_result is None or count < best_count:
                best_count = count
                best_result = improved
    
    # Final fallback: random search
    if best_result is None:
        # Just return 12 random small numbers
        best_result = list(range(1, 13))
    
    return best_result
