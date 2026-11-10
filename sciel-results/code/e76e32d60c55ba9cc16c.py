"""Construct the largest admissible tuple within [0, 5000]."""
import time
import random

N = 5000
DEADLINE = 330.0
SAFETY_MARGIN = 5.0  # seconds before the kill deadline to stop and return

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i, v in enumerate(sieve) if v]

def _is_admissible(points, primes):
    """Check if points form an admissible tuple for the given primes."""
    n = len(points)
    point_set = set(points)
    for p in primes:
        if p > n:
            break
        # Check if all p residue classes mod p are covered
        covered = set()
        for v in points:
            covered.add(v % p)
        if len(covered) == p:
            return False
    return True

def construct():
    start_time = time.time()
    
    # Precompute primes up to a reasonable bound
    # The tuple length will be at most N+1 = 5001, so we need primes up to ~5001
    # But for efficiency, let's compute primes up to 5001
    max_len = N + 1
    primes = _sieve_primes(max_len)
    
    # Strategy: Start with a large candidate set and iteratively remove points
    # to make the tuple admissible while keeping it as large as possible.
    
    # A better approach: Start with all points [0, N] and remove points to make it admissible.
    # But that's too large. Let's think differently.
    
    # The seed program starts with 1 and primes above sqrt(N), then repairs.
    # Let's try a more systematic approach:
    
    # Key insight: For the tuple to be admissible, for each prime p <= len(tuple),
    # we must miss at least one residue class mod p.
    
    # Strategy: Build the tuple greedily, adding points that don't violate admissibility.
    # Or: Start with a large set and remove minimally.
    
    # Let's use a repair-based approach with smart initialization:
    # 1. Start with a candidate set that's likely to be close to admissible
    # 2. For each prime p <= len(candidate), if all p classes are covered,
    #    remove the point from the least-populated class
    
    # Better initialization: use a dense set but structured to miss classes
    # For each prime p, we want to miss one class. Let's choose class 0 for each p.
    # So we want points that are NOT congruent to 0 mod p for each p <= len(tuple).
    # But that's too restrictive.
    
    # Let's try: start with all integers in [0, N], then for each prime p <= current_len,
    # if all p classes are covered, remove points from the least-populated class.
    # This should converge to a large admissible tuple.
    
    # Even better: start with a subset that's already partially admissible.
    # Use the seed's idea but improve it.
    
    # Let's try a hybrid: start with primes > sqrt(N) plus 1, then add more points
    # while maintaining admissibility.
    
    def _try_add(points, primes_list, candidate):
        """Check if adding candidate to points keeps the tuple admissible."""
        new_points = points + [candidate]
        n = len(new_points)
        for p in primes_list:
            if p > n:
                break
            covered = set()
            for v in new_points:
                covered.add(v % p)
            if len(covered) == p:
                return False
        return True
    
    # Approach: Start with a good seed, then greedily add points
    # that maintain admissibility.
    
    # Seed: 1 and primes > sqrt(N)
    sqrt_n = int(N ** 0.5) + 1
    seed_points = [1] + [p for p in primes if p > sqrt_n and p <= N]
    
    # Sort remaining candidates by some heuristic
    # Try to add points that are "spread out" in residue classes
    remaining = [v for v in range(N + 1) if v not in set(seed_points)]
    
    # For efficiency, we'll use a repair approach instead:
    # Start with a large random subset, then repair to admissibility.
    
    # Let's use the repair approach with a better initial set:
    # Start with all points, then remove to make admissible.
    # But that's O(N * num_primes * len) which might be too slow.
    
    # Alternative: Use a local search / simulated annealing approach.
    # Start with the seed, then try to add points one at a time.
    
    # For speed, let's use a bitset-like approach for residue tracking.
    
    # Let's go with the repair approach but with a smarter initial set:
    # Start with a dense set, then repair.
    
    # Initialize with a random subset of size ~N/2, then repair.
    # Actually, let's just use the seed's repair approach but with more iterations
    # and a better removal strategy.
    
    points = list(seed_points)
    point_set = set(points)
    
    # Try to add points greedily
    random.seed(42)
    random.shuffle(remaining)
    
    for candidate in remaining:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        if _try_add(points, primes, candidate):
            points.append(candidate)
            point_set.add(candidate)
    
    # Now repair: for any prime p <= len(points) where all classes are covered,
    # remove from the least-populated class
    changed = True
    while changed:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        changed = False
        n = len(points)
        for p in primes:
            if p > n:
                break
            # Count residue classes
            class_counts = {}
            for v in points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            if len(class_counts) == p:
                # All classes covered, remove from least-populated
                min_class = min(class_counts, key=class_counts.get)
                # Remove one point from this class
                # Prefer to remove the largest point to keep the set "dense" at low values
                candidates_to_remove = [v for v in points if v % p == min_class]
                # Remove the largest one
                candidates_to_remove.sort(reverse=True)
                to_remove = candidates_to_remove[0]
                points.remove(to_remove)
                point_set.discard(to_remove)
                changed = True
                break
    
    # Verify admissibility
    if not _is_admissible(points, primes):
        # Fallback: use the seed's result
        points = [1] + [p for p in primes if p * p > N and p <= N]
    
    return points

# Ensure we always return within deadline
def construct():
    start_time = time.time()
    max_time = DEADLINE - SAFETY_MARGIN
    
    # Precompute primes
    max_len = N + 1
    primes = _sieve_primes(max_len)
    
    # Seed initialization
    sqrt_n = int(N ** 0.5) + 1
    points = [1] + [p for p in primes if p > sqrt_n and p <= N]
    point_set = set(points)
    
    # Greedy addition
    remaining = [v for v in range(N + 1) if v not in point_set]
    random.seed(42)
    random.shuffle(remaining)
    
    def _can_add(candidate, current_points, primes_list):
        n = len(current_points) + 1
        for p in primes_list:
            if p > n:
                break
            # Check if adding candidate would cover all p classes
            covered = set()
            for v in current_points:
                covered.add(v % p)
            if candidate % p not in covered:
                covered.add(candidate % p)
            if len(covered) == p:
                return False
        return True
    
    for candidate in remaining:
        if time.time() - start_time > max_time:
            break
        if _can_add(candidate, points, primes):
            points.append(candidate)
            point_set.add(candidate)
    
    # Repair phase
    changed = True
    while changed:
        if time.time() - start_time > max_time:
            break
        changed = False
        n = len(points)
        for p in primes:
            if p > n:
                break
            class_counts = {}
            for v in points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            if len(class_counts) == p:
                min_class = min(class_counts, key=class_counts.get)
                candidates_to_remove = [v for v in points if v % p == min_class]
                candidates_to_remove.sort(reverse=True)
                to_remove = candidates_to_remove[0]
                points.remove(to_remove)
                point_set.discard(to_remove)
                changed = True
                break
    
    # Final verification and fallback
    if not _is_admissible(points, primes):
        points = [1] + [p for p in primes if p * p > N and p <= N]
    
    return points
