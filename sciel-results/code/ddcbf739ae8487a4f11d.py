"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time
import random

_DEADLINE = 320.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple: for every prime p in primes_to_check,
    some residue class mod p is missed."""
    point_set = set(points)
    for p in primes_to_check:
        seen = set()
        for v in point_set:
            seen.add(v % p)
        if len(seen) == p:
            return False
    return True


def _get_covered_primes(points, primes_to_check):
    """Return list of primes p where all p residue classes are covered."""
    point_set = set(points)
    covered = []
    for p in primes_to_check:
        seen = set()
        for v in point_set:
            seen.add(v % p)
        if len(seen) == p:
            covered.append(p)
    return covered


def _greedy_removal(points, primes_to_check, deadline):
    """Greedy removal: repeatedly find a covered prime, remove least-populated class."""
    points = list(points)
    point_set = set(points)
    
    while time.time() < deadline:
        # Find all covered primes
        covered = []
        for p in primes_to_check:
            if p > len(point_set):
                continue  # Can't cover all p classes with fewer than p elements
            seen = set()
            for v in point_set:
                seen.add(v % p)
            if len(seen) == p:
                covered.append(p)
        
        if not covered:
            break
        
        # Pick the smallest covered prime (most constrained)
        p = min(covered)
        
        # Count elements in each residue class mod p
        class_counts = {}
        for v in point_set:
            r = v % p
            class_counts[r] = class_counts.get(r, 0) + 1
        
        # Remove the least populated class
        min_class = min(class_counts, key=lambda r: class_counts[r])
        to_remove = [v for v in point_set if v % p == min_class]
        
        for v in to_remove:
            point_set.discard(v)
        
        points = list(point_set)
    
    return points


def _add_back_search(points, primes_to_check, deadline):
    """Try to add back removed elements to increase tuple size while maintaining admissibility."""
    points = list(points)
    point_set = set(points)
    all_candidates = set(range(N + 1)) - point_set
    
    # Precompute primes we need to check - only those <= len(point_set) + 1 matter
    # because we need at least p elements to cover all p residue classes
    def _get_check_primes(current_size):
        # We need p <= current_size + 1 for it to be possible to cover all p classes
        # But we should check p <= current_size as well since adding one element could make it p
        return [p for p in primes_to_check if p <= current_size + 1]
    
    check_primes = _get_check_primes(len(point_set))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort candidates by how "safe" they are - prefer those that don't fill any residue class
        candidates = list(all_candidates)
        
        # For each candidate, compute how many new residue classes it would fill
        # for each prime - prefer candidates that fill fewer new classes
        def _safety_score(v):
            score = 0
            for p in check_primes:
                r = v % p
                # Check if this residue class already exists in point_set
                exists = any(x % p == r for x in point_set)
                if not exists:
                    score += 1  # This would be the first element in this class, risky
            return score
        
        # Sort by safety score (lower is safer)
        candidates.sort(key=_safety_score)
        
        for v in candidates:
            if time.time() >= deadline:
                break
            # Try adding v
            # Check admissibility for all relevant primes
            admissible = True
            for p in check_primes:
                seen = set()
                for x in point_set:
                    seen.add(x % p)
                if len(seen) == p - 1 and v % p not in seen:
                    # Adding v would complete all p classes
                    admissible = False
                    break
                elif len(seen) == p:
                    admissible = False
                    break
            
            if admissible:
                point_set.add(v)
                all_candidates.discard(v)
                improved = True
                # Update check_primes if needed
                check_primes = _get_check_primes(len(point_set))
    
    return list(point_set)


def _initial_full_set():
    """Create the full set [0, N] - start with everything and remove to make admissible."""
    return list(range(N + 1))


def _initial_random_set(fraction=0.5, seed=None):
    """Create a random initial set of given fraction of [0, N]."""
    if seed is not None:
        random.seed(seed)
    return random.sample(range(N + 1), int(fraction * (N + 1)))


def _full_pipeline(points, deadline):
    """Run greedy removal then add-back search."""
    primes_to_check = _primes(N + 1)
    
    # Phase 1: Greedy removal to make admissible
    points = _greedy_removal(points, primes_to_check, deadline)
    
    # Phase 2: Add-back search to maximize size
    points = _add_back_search(points, primes_to_check, deadline)
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Get primes up to N+1 for checking
    all_primes = _primes(N + 1)
    
    best = []
    
    # Strategy 1: Start from full set [0, N] and sieve down - this is the key insight
    # from the inspiration: sieving from full integer set finds denser admissible sets
    if time.time() < deadline - 10:
        init_full = _initial_full_set()
        result_full = _full_pipeline(init_full, deadline)
        if len(result_full) > len(best):
            best = result_full
    
    # Strategy 2: If we have time, try random restarts with different densities
    attempts = 0
    while time.time() < deadline - 15 and attempts < 5:
        # Vary the fraction for diversity
        fraction = 0.3 + 0.4 * (attempts % 4) / 4.0  # 0.3, 0.4, 0.5, 0.6
        seed = attempts * 7919 + 12345
        init = _initial_random_set(fraction=fraction, seed=seed)
        result = _full_pipeline(init, deadline)
        if len(result) > len(best):
            best = result
        attempts += 1
    
    # Final: if we have time, do one more add-back on the best
    if time.time() < deadline - 5 and best:
        best = _add_back_search(best, all_primes, deadline)
    
    return best
# EVOLVE-BLOCK-END
