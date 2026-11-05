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

def _generate_smooth_numbers(prime_list, max_val):
    """Generate all S-smooth numbers up to max_val using BFS/DFS."""
    if not prime_list:
        return set()
    smooth = set([1])
    for p in prime_list:
        new_smooth = set()
        for s in smooth:
            power = p
            while s * power <= max_val:
                new_smooth.add(s * power)
                power *= p
        smooth.update(new_smooth)
    return smooth

def _find_clique(adj, candidates, target_size, deadline):
    """Find a clique of target_size in the graph defined by adj and candidates."""
    n = len(candidates)
    if n < target_size:
        return None
    
    # Greedy: pick vertices with highest degree first
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Try to build clique greedily
    clique = []
    remaining = set(order)
    
    while len(clique) < target_size and remaining:
        # Pick vertex with max degree among remaining
        best_v = max(remaining, key=lambda v: len(adj[v] & remaining))
        
        # Check if best_v is compatible with all in clique
        compatible = True
        for c in clique:
            if c not in adj[best_v]:
                compatible = False
                break
        
        if compatible:
            clique.append(best_v)
            remaining.discard(best_v)
            # Remove vertices not compatible with new clique member
            to_remove = []
            for v in remaining:
                if v not in adj[best_v]:
                    to_remove.append(v)
            for v in to_remove:
                remaining.discard(v)
        else:
            remaining.discard(best_v)
    
    if len(clique) >= target_size:
        return [candidates[i] for i in clique[:target_size]]
    return None

def _find_12_smooth_sumset(prime_set, limit=10000, deadline=None):
    """
    Find 12 distinct positive integers <= limit such that all pairwise sums
    are composed only of primes in prime_set.
    """
    if deadline and time.time() > deadline:
        return None
    
    # Generate all S-smooth numbers up to 2*limit (max possible sum)
    smooth_sums = _generate_smooth_numbers(prime_set, 2 * limit)
    
    # A number a is a candidate if there exist at least 11 other numbers b
    # such that a+b is S-smooth
    # For each a, compatible b's are: {s - a : s in smooth_sums, 1 <= s-a <= limit, s-a != a}
    
    # Build candidate list and their compatibility sets
    # To be efficient, only consider a's that have many compatible partners
    
    # First pass: count compatibility for each a
    compat_count = {}
    for a in range(1, limit + 1):
        if deadline and time.time() > deadline:
            return None
        count = 0
        for s in smooth_sums:
            b = s - a
            if 1 <= b <= limit and b != a:
                count += 1
        if count >= 11:
            compat_count[a] = count
    
    if len(compat_count) < 12:
        return None
    
    # Sort candidates by compatibility count
    sorted_candidates = sorted(compat_count.items(), key=lambda x: -x[1])
    
    # Take top candidates (limit to keep search tractable)
    top_n = min(100, len(sorted_candidates))
    top_candidates = [c[0] for c in sorted_candidates[:top_n]]
    
    # Build adjacency: two candidates a, b are adjacent if a+b is in smooth_sums
    n_cand = len(top_candidates)
    adj = [set() for _ in range(n_cand)]
    
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if top_candidates[i] + top_candidates[j] in smooth_sums:
                adj[i].add(j)
                adj[j].add(i)
    
    # Find clique of size 12
    result = _find_clique(adj, top_candidates, 12, deadline)
    if result:
        return sorted(result)
    
    return None

def _local_search_improve(initial_set, limit=10000, deadline=None):
    """
    Local search to improve a set by replacing elements.
    Uses fast prime counting.
    """
    current = sorted(initial_set)
    current_count, _ = _count_distinct_primes(current)
    
    best = current.copy()
    best_count = current_count
    
    # Precompute factorizations for all possible sums up to 2*limit
    # This is expensive but done once
    max_sum = 2 * limit
    prime_factors_cache = {}
    
    def get_prime_factors(n):
        if n in prime_factors_cache:
            return prime_factors_cache[n]
        pf = _factorize(n)
        prime_factors_cache[n] = pf
        return pf
    
    # Generate candidate replacements: focus on small numbers and smooth numbers
    candidates = list(range(1, min(200, limit + 1)))
    # Add some smooth numbers
    for p2 in range(1, 14):
        for p3 in range(1, 14):
            v = (2 ** p2) * (3 ** p3)
            if v <= limit and v not in candidates:
                candidates.append(v)
    candidates = sorted(set(candidates))
    
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
        
        # Count primes efficiently
        prime_set = set()
        for i in range(12):
            for j in range(i + 1, 12):
                prime_set.update(get_prime_factors(new_set[i] + new_set[j]))
        
        new_count = len(prime_set)
        
        if new_count < best_count:
            best_count = new_count
            best = new_set
            current = new_set
            current_count = new_count
        elif new_count == best_count and random.random() < 0.1:
            current = new_set
            current_count = new_count
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # Strategy: Try different small prime sets and find 12 numbers where
    # all pairwise sums are S-smooth for that set S
    
    # Start with very small prime sets
    prime_sets_to_try = [
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
            improved, count = _local_search_improve(initial, 10000, deadline)
            
            if best_result is None or count < best_count:
                best_count = count
                best_result = improved
    
    # Final fallback: random search
    if best_result is None:
        # Just return 12 random small numbers
        best_result = list(range(1, 13))
    
    return best_result
