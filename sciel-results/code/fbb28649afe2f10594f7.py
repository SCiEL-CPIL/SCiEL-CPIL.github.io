"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000
_START_TIME = time.time()
_TIME_LIMIT = 280  # wall-clock safety margin before 300s kill


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


def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple for the given primes."""
    n = len(points)
    if n == 0:
        return True
    point_set = set(points)
    for p in primes_to_check:
        if p > n:
            break
        # Check if all residue classes mod p are covered
        covered = set(v % p for v in point_set)
        if len(covered) == p:
            return False
    return True


def _repair_admissible(points, primes_to_check):
    """Repair points to be admissible by removing least-populated residue classes."""
    points = list(points)
    point_set = set(points)
    changed = True
    while changed:
        changed = False
        for p in primes_to_check:
            if p > len(points):
                break
            # Count occurrences of each residue class
            class_counts = {}
            for v in point_set:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            
            if len(class_counts) == p:
                # All residue classes covered; remove the least-populated one
                min_class = min(class_counts, key=class_counts.get)
                removed = [v for v in point_set if v % p == min_class]
                for v in removed:
                    point_set.discard(v)
                points = list(point_set)
                changed = True
                break
    return points


def construct():
    """Construct the largest admissible tuple within [0, N]."""
    primes_N = _primes(N)
    
    # Strategy: start from all integers [0, N] and repair
    # This retains more elements before forced removals
    points = list(range(N + 1))
    
    # Get primes up to N (for checking admissibility of the full set)
    # Actually, for admissibility we only need primes p <= len(tuple)
    # But we also need to ensure the set is "admissible" in the sense that
    # for every prime p <= len(tuple), some residue class mod p is missed
    
    # Repair: for each prime p <= len(points), if all p residue classes are covered,
    # remove the least-populated class
    primes_to_check = _primes(N)  # primes up to N, we'll filter by length
    
    # Iteratively repair
    points = _repair_admissible(points, primes_to_check)
    
    # Now try to maximize further by considering which removals were necessary
    # and whether we can recover some elements
    
    # Alternative approach: greedy construction
    # Start with empty set, add elements one by one if they don't violate admissibility
    # But this is slow for N=5000
    
    # Better: use the repair result, then try local optimization
    # For each element removed during repair, check if adding it back maintains admissibility
    # Actually, we can't easily recover removed elements
    
    # Let's try a different strategy:
    # 1. Start with [0, N]
    # 2. For each prime p <= N, determine which residue classes to keep
    # 3. Keep only elements that are in at least one "safe" residue class for each prime
    
    # More principled approach:
    # The admissibility condition is: for each prime p <= m (where m = len(tuple)),
    # the tuple does NOT cover all p residue classes mod p.
    
    # This is a constraint satisfaction problem. Let's use a greedy + local search approach.
    
    # First, let's see what the repair gave us
    current_size = len(points)
    
    # Try to improve by attempting to add back removed elements
    # We need to track which elements were removed
    # Actually, let's restart with a cleaner approach
    
    # Clean approach:
    # 1. Start with all of [0, N]
    # 2. For primes in increasing order p = 2, 3, 5, ...:
    #    - If p > len(current_set), stop
    #    - Check if current_set covers all p residue classes mod p
    #    - If yes, remove elements from the least-populated residue class
    # 3. Return the resulting set
    
    # This is essentially what _repair_admissible does, but let's be more careful
    
    # Restart
    points = list(range(N + 1))
    point_set = set(points)
    
    # Process primes in increasing order
    for p in primes_to_check:
        if p > len(point_set):
            break
        
        # Count residue classes
        class_counts = {}
        for v in point_set:
            r = v % p
            class_counts[r] = class_counts.get(r, 0) + 1
        
        if len(class_counts) == p:
            # All classes covered; remove least-populated
            min_class = min(class_counts, key=class_counts.get)
            to_remove = [v for v in point_set if v % p == min_class]
            for v in to_remove:
                point_set.discard(v)
    
    points = sorted(point_set)
    result_size = len(points)
    
    # Now try to add back elements that were removed, if possible
    # Build a set of all elements for reference
    all_elements = set(range(N + 1))
    current_set = set(points)
    removed_elements = all_elements - current_set
    
    # Try to add back removed elements one at a time
    # For efficiency, only try if adding doesn't violate admissibility
    # Check admissibility after each addition
    
    primes_for_check = _primes(result_size + 1)  # primes up to current size
    
    improved = True
    while improved:
        improved = False
        for elem in sorted(removed_elements):
            if time.time() - _START_TIME > _TIME_LIMIT - 10:
                break
            
            # Try adding elem
            test_set = current_set | {elem}
            test_size = len(test_set)
            
            # Check admissibility
            is_ok = True
            for p in primes_for_check:
                if p > test_size:
                    break
                covered = set(v % p for v in test_set)
                if len(covered) == p:
                    is_ok = False
                    break
            
            if is_ok:
                current_set.add(elem)
                removed_elements.discard(elem)
                improved = True
                # Update primes for check since size increased
                new_size = len(current_set)
                primes_for_check = _primes(new_size + 1)
    
    # Final cleanup: ensure admissibility
    final_points = sorted(current_set)
    
    # Double-check admissibility
    final_size = len(final_points)
    primes_final = _primes(final_size)
    for p in primes_final:
        covered = set(v % p for v in final_points)
        if len(covered) == p:
            # Need to fix: remove least-populated class
            class_counts = {}
            for v in final_points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            min_class = min(class_counts, key=class_counts.get)
            final_points = [v for v in final_points if v % p != min_class]
    
    return final_points
