"""
Narrow admissible tuple construction.
Combines greedy removal with an optimized, incremental add-back local search.
The key improvement over the previous version is the use of pre-computed residue
counts and a fast "can_add" check that avoids O(N) scans for every candidate.
"""

import time
import random
import numpy as np

N = 5000
DEADLINE = 330.0  # wall-clock seconds allowed by evaluator; keep safety margin
SAFETY_MARGIN = 15.0  # return best result before this margin


def _primes(limit):
    """Sieve of Eratosthenes."""
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


def _is_admissible(points, primes_to_check):
    """Check if points is admissible: for every prime p <= len(points), some residue class mod p is missed."""
    if not points:
        return True
    n = len(points)
    for p in primes_to_check:
        if p > n:
            break
        residues = set(x % p for x in points)
        if len(residues) == p:
            return False
    return True


def _greedy_construct(seed=0):
    """
    Greedy construction: start with all points, remove elements from the most
    populated residue class for each prime that is fully covered.
    Uses randomized tie-breaking for variety.
    """
    rng = random.Random(seed)
    points = list(range(N + 1))
    
    changed = True
    while changed:
        changed = False
        current_len = len(points)
        # Only need to check primes <= current_len
        for p in _ALL_PRIMES:
            if p > current_len:
                break
            # Count residues
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            if all(c > 0 for c in counts):
                # Find the smallest class to remove
                min_count = min(counts)
                # Randomly pick one of the smallest classes to break symmetry
                candidates = [i for i, c in enumerate(counts) if c == min_count]
                min_idx = rng.choice(candidates)
                # Remove all points in that class
                points = [v for v in points if v % p != min_idx]
                changed = True
                break
    
    return points


def _local_search_add(points, seed=0, max_rounds=20):
    """
    Optimized local search: try to add back removed elements one at a time if it doesn't violate admissibility.
    Uses incremental residue counting and a fast check for validity.
    """
    rng = random.Random(seed)
    points_set = set(points)
    candidates = [x for x in range(N + 1) if x not in points_set]
    rng.shuffle(candidates)
    
    n = len(points)
    # We need to track residue counts for all primes p <= n (and potentially n+1)
    # Let's determine the relevant primes
    # Since n can grow, we should be careful. But p <= n is the constraint.
    # If we add an element, n becomes n+1. We need to check primes <= n+1.
    
    # Precompute residue counts for current points for all primes up to N+1?
    # That's too expensive.
    # Instead, we maintain counts for primes p <= current_n.
    # When we add an element, we update counts. If n increases, we might need to add counts for new primes.
    
    # Let's use a dictionary: p -> list of counts
    residue_counts = {}
    for p in _ALL_PRIMES:
        if p > n:
            break
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
    
    # Also track the number of missing residues for each prime to speed up the check
    # missing[p] = number of residues r in [0, p-1] such that counts[r] == 0
    missing_counts = {}
    for p, counts in residue_counts.items():
        missing_counts[p] = sum(1 for c in counts if c == 0)
    
    improved = False
    for round_num in range(max_rounds):
        if not candidates:
            break
        # Shuffle candidates again for each round? Or just iterate?
        # Shuffling helps avoid bias.
        rng.shuffle(candidates)
        
        added_in_round = False
        new_candidates = []
        
        for x in candidates:
            # Check if adding x keeps admissibility
            new_n = n + 1
            can_add = True
            
            # We need to check all primes p <= new_n
            # For primes p <= n, we have counts.
            # For p = new_n (if prime), we need to check if adding x completes the set.
            # Actually, if p <= n, we have the state.
            # If p > n, we don't have state, but if p > n, then p > len(points) currently.
            # The condition is: for every prime p <= len(tuple), some class is missed.
            # So if we add x, len becomes new_n. We must check primes p <= new_n.
            
            # Iterate through primes <= new_n
            for p in _ALL_PRIMES:
                if p > new_n:
                    break
                
                if p in residue_counts:
                    counts = residue_counts[p]
                    missing = missing_counts[p]
                    r = x % p
                    
                    # If adding x makes a residue class that was 0 become 1, and that was the only missing one,
                    # then missing becomes 0, which is invalid.
                    if counts[r] == 0:
                        if missing == 1:
                            can_add = False
                            break
                    # If counts[r] > 0, missing doesn't change.
                    # We don't need to update anything if we fail.
                    # If we succeed, we update later.
                else:
                    # This prime p was not in our tracked set because p > n (old size).
                    # This implies p == new_n (since p is the next prime).
                    # We need to check if adding x completes the residue set mod p.
                    # Since p > n, and we have n points, we have at most n residues.
                    # Since p is prime and p > n, p > n >= 1.
                    # Actually, if p > n, it's impossible to have all p residues covered with n points.
                    # Wait, the condition is "some residue class is missed".
                    # If we have n points and p > n, we can cover at most n residues.
                    # So at least p - n residues are missed. Since p > n, p - n >= 1.
                    # So it is ALWAYS admissible for primes p > n.
                    # Therefore, we only need to check primes p <= n.
                    # The new prime p = new_n doesn't need to be checked for violation?
                    # Let's re-read: "for every prime p <= len(tuple) some residue class mod p is missed".
                    # If len(tuple) = new_n, and p = new_n, we need a missed class mod new_n.
                    # With new_n points, can we cover all new_n residues?
                    # Yes, if the points are distinct mod new_n.
                    # But new_n is prime.
                    # If we have new_n distinct integers, they are NOT necessarily distinct mod new_n.
                    # Example: N=5, points {0, 1, 2, 3, 4}. len=5. p=5. Residues {0,1,2,3,4}. All covered. Not admissible.
                    # So we DO need to check p = new_n.
                    
                    # To check p = new_n efficiently:
                    # We need to know if the set of residues of (points + [x]) mod p has size p.
                    # Since p is large (approx N), and we have N points, this is expensive.
                    # However, this case only happens when we increase the size.
                    # And p is the next prime.
                    # How often does this happen?
                    # Only when we successfully add an element.
                    # So we can do a full check for this specific prime if needed.
                    
                    # But wait, if p > n, we didn't track counts.
                    # Let's just do a full check for p if p is not in residue_counts.
                    # This is rare.
                    
                    residues = set(v % p for v in points)
                    residues.add(x % p)
                    if len(residues) == p:
                        can_add = False
                        break
            
            if can_add:
                # Add x
                points.append(x)
                points_set.add(x)
                n = new_n
                
                # Update residue counts
                for p in _ALL_PRIMES:
                    if p > n:
                        break
                    if p in residue_counts:
                        counts = residue_counts[p]
                        r = x % p
                        if counts[r] == 0:
                            missing_counts[p] -= 1
                        counts[r] += 1
                    else:
                        # Initialize counts for this new prime p
                        counts = [0] * p
                        for v in points:
                            counts[v % p] += 1
                        residue_counts[p] = counts
                        missing_counts[p] = sum(1 for c in counts if c == 0)
                
                added_in_round = True
                # Remove x from candidates
                # Instead of removing from list, we just don't add to new_candidates
            else:
                new_candidates.append(x)
        
        if not added_in_round:
            break
        candidates = new_candidates
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_points = None
    best_size = 0
    
    # Strategy 1: Greedy construction with different seeds
    # The greedy construction is fast. We can run many seeds.
    # Each seed produces a different initial admissible set.
    # Then we run local search to improve it.
    
    num_seeds = 200
    for seed in range(num_seeds):
        if time.time() > deadline:
            break
        
        points = _greedy_construct(seed=seed)
        
        # Local search
        points = _local_search_add(points, seed=seed * 1000, max_rounds=50)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = list(points)
            
            # If we get a very large set, we might stop early?
            # No, keep trying to beat it.
    
    # Strategy 2: If we have time, try to polish the best result
    if best_points is not None and time.time() < deadline - 10:
        # Run more local search rounds on the best result
        # We need to be careful not to degrade it.
        # _local_search_add only adds points, so it's monotonic in size.
        # It won't make it worse.
        for extra_seed in range(100):
            if time.time() > deadline:
                break
            best_points = _local_search_add(best_points, seed=9999 + extra_seed, max_rounds=100)
    
    # Fallback
    if best_points is None:
        # This should not happen, but just in case
        best_points = [0]
    
    # Final verification
    # Ensure distinct integers in [0, N]
    best_points = list(set(best_points))
    best_points = [x for x in best_points if 0 <= x <= N]
    
    # Verify admissibility
    p_limit = len(best_points)
    primes_to_check = [p for p in _ALL_PRIMES if p <= p_limit]
    if not _is_admissible(best_points, primes_to_check):
        # Repair by removing minority classes
        changed = True
        while changed:
            changed = False
            current_len = len(best_points)
            for p in _ALL_PRIMES:
                if p > current_len:
                    break
                counts = [0] * p
                for v in best_points:
                    counts[v % p] += 1
                if all(c > 0 for c in counts):
                    min_idx = min(range(p), key=lambda i: counts[i])
                    best_points = [v for v in best_points if v % p != min_idx]
                    changed = True
    
    return best_points


# EVOLVE-BLOCK-START
# (The above functions are the actual implementation; this block is for structural compatibility)
# EVOLVE-BLOCK-END
