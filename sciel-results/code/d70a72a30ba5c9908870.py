"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to N+1
_ALL_PRIMES = _primes(N + 1)
# Precompute prime lists for various sizes
_PRIME_LISTS = {}
for _sz in range(0, N + 2, 10):
    _PRIME_LISTS[_sz] = _primes(_sz)

def _primes_up_to(n):
    """Get primes up to n using cache."""
    # Find the largest cached size >= n
    for sz in range(n, N + 2, 10):
        if sz in _PRIME_LISTS:
            return [p for p in _PRIME_LISTS[sz] if p <= n]
    return _primes(n)

def _is_admissible_fast(points, primes):
    """Check admissibility. primes: list of primes to check (should be primes <= len(points))."""
    n = len(points)
    if n == 0:
        return True
    for p in primes:
        if p > n:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
        if len(residues) == p:
            return False
    return True

def _greedy_remove(points_set, rng, time_start, time_limit):
    """Greedy removal. Returns (remaining_set, removed_set)."""
    points = set(points_set)
    removed = set()
    
    while time.time() - time_start < time_limit:
        n = len(points)
        if n == 0:
            break
        ps = _primes_up_to(n)
        rng.shuffle(ps)
        
        found_violation = False
        for p in ps:
            if p > n:
                continue
            # Count residues
            residue_counts = {}
            for v in points:
                r = v % p
                residue_counts[r] = residue_counts.get(r, 0) + 1
            
            if len(residue_counts) == p:
                # Violation: all residues present
                min_count = min(residue_counts.values())
                candidates = [r for r, c in residue_counts.items() if c == min_count]
                chosen = rng.choice(candidates)
                to_remove = {v for v in points if v % p == chosen}
                points -= to_remove
                removed |= to_remove
                found_violation = True
                break
        
        if not found_violation:
            break
    
    return points, removed

def _add_back(points_set, removed_set, rng, time_start, time_limit):
    """Try to add back removed elements."""
    points = set(points_set)
    
    if not removed_set:
        return points
    
    rem_list = list(removed_set)
    rng.shuffle(rem_list)
    
    for v in rem_list:
        if time.time() - time_start > time_limit:
            break
        if v in points:
            continue
        
        # Quick check: try adding v
        new_points = points | {v}
        n = len(new_points)
        ps = _primes_up_to(n)
        
        ok = True
        for p in ps:
            if p > n:
                break
            residues = set()
            for x in new_points:
                residues.add(x % p)
                if len(residues) == p:
                    ok = False
                    break
            if not ok:
                break
        
        if ok:
            points = new_points
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    best_result = None
    best_size = -1
    
    # Define various initial sets
    seed_set = set([1]) | set(p for p in _ALL_PRIMES if p * p > N)
    
    rng_master = random.Random(42)
    random_sets = []
    for i in range(5):
        size = 1000 + i * 500  # 1000, 1500, 2000, 2500, 3000
        random_sets.append(set(rng_master.sample(range(N + 1), size)))
    
    full_set = set(range(N + 1))
    
    initial_sets = [seed_set] + random_sets + [full_set]
    
    attempt = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Time budget for this attempt
        t_budget = min(remaining * 0.85, 15)
        
        # Select initial set
        init_set = initial_sets[attempt % len(initial_sets)]
        
        rng = random.Random(attempt * 7919 + 12345)
        
        t0 = time.time()
        # Greedy removal
        points, removed = _greedy_remove(init_set, rng, t0, t_budget * 0.5)
        
        # Add back
        if removed and time.time() - t0 < t_budget * 0.8:
            points = _add_back(points, removed, rng, t0, t_budget)
        
        result = list(points)
        size = len(result)
        
        if size > best_size:
            best_size = size
            best_result = result
        
        attempt += 1
    
    if best_result is None:
        # Fallback to seed
        points = [1] + [p for p in _ALL_PRIMES if p * p > N]
        changed = True
        while changed:
            changed = False
            for p in _primes(len(points)):
                classes = {}
                for v in points:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    points = [v for v in points if v not in smallest]
                    changed = True
                    break
        best_result = points
    
    return best_result
# EVOLVE-BLOCK-END
