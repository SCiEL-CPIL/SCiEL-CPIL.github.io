import time
import numpy as np
import random
import sys

N = 5000
RANGE = N + 1

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

# Precompute primes up to N
PRIMES = _primes(N)

def _is_admissible(points, primes):
    """Check if a set of points is admissible."""
    n = len(points)
    if n == 0:
        return True
    # points is a list of ints
    point_set = set(points)
    for p in primes:
        if p > n:
            break
        # Check if all residue classes mod p are covered
        residues = set(x % p for x in point_set)
        if len(residues) == p:
            return False
    return True

def _greedy_remove_min(points, primes, seed, deadline):
    """
    Greedy removal: for each prime p <= len(points), remove the residue class
    with the minimum count. Tie-breaking is randomized using seed.
    Returns list of remaining points.
    """
    rng = random.Random(seed)
    active = np.ones(RANGE, dtype=bool)
    current_size = RANGE
    
    # We iterate through primes. Note: as we remove points, len(points) decreases.
    # The condition is p <= current_size.
    # We process primes in increasing order.
    for p in primes:
        if time.time() > deadline:
            break
        if p > current_size:
            # Since primes are sorted, if p > current_size, all subsequent primes are also > current_size
            # (unless current_size was very large and we skipped some? No, p increases, current_size decreases or stays same)
            # Actually, if p > current_size, we can break because the condition is p <= len(tuple).
            # Wait, the condition is "for every prime p <= len(tuple)". 
            # If current_size is 100, and we are at p=101, we don't need to process p=101.
            # So we can break.
            break
        
        # Get active indices
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        # Count residues
        residues = active_indices % p
        counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(counts)
        
        if min_count == 0:
            # Already admissible for this prime, do nothing
            continue
            
        # Find all classes with min count
        candidate_classes = np.where(counts == min_count)[0]
        
        # Pick one randomly
        r = candidate_classes[rng.randint(0, len(candidate_classes) - 1)]
        
        # Remove active points in this residue class
        # Mask: x % p == r AND active
        mask = (np.arange(RANGE) % p == r) & active
        removed = np.sum(mask)
        if removed > 0:
            active = active & ~mask
            current_size -= removed
            
    return np.where(active)[0].tolist()

def _greedy_remove_max(points, primes, seed, deadline):
    """
    Greedy removal: for each prime p <= len(points), remove the residue class
    with the MAXIMUM count. This is counter-intuitive but might lead to different structures.
    Actually, removing max count removes more points, leading to smaller tuple.
    We want largest tuple. So min count is the standard approach.
    Let's stick to min count but maybe vary the order or tie-breaking.
    """
    # This function is likely not useful for maximizing size, but included for structure if needed.
    # For now, let's just call _greedy_remove_min.
    return _greedy_remove_min(points, primes, seed, deadline)

def _polish(points, primes, deadline):
    """
    Try to add back points that were removed, if possible, to increase size.
    This is a local search improvement.
    """
    point_set = set(points)
    current_size = len(points)
    
    # Candidates are points in [0, N] not in point_set
    candidates = [x for x in range(RANGE) if x not in point_set]
    
    # Shuffle candidates to try in random order
    random.shuffle(candidates)
    
    improved = True
    while improved:
        improved = False
        for x in candidates:
            if time.time() > deadline:
                return points
            
            # Try adding x
            new_set = list(point_set) + [x]
            new_size = len(new_set)
            
            # Check admissibility for the new set
            # Optimization: only check primes that might be violated.
            # Adding x might complete a residue class for some p <= new_size.
            # Specifically, for any p <= new_size, if x % p was the missing residue, it's now violated.
            # We can check all p <= new_size.
            
            valid = True
            for p in primes:
                if p > new_size:
                    break
                # Check if all residues are covered in new_set
                # We can maintain counts, but for simplicity and speed in Python loop,
                # let's just check the specific prime p.
                # Actually, re-checking all primes is O(P * N) which is slow.
                # Better: check if x completes a class.
                # For a given p, let missing_residues_p be the set of residues not in {y % p for y in point_set}.
                # If x % p is in missing_residues_p, then adding x covers it.
                # If missing_residues_p was empty (i.e., all covered), then adding x doesn't change validity for p.
                # Wait, if missing_residues_p was empty, the set was NOT admissible for p.
                # But we assume point_set is admissible. So missing_residues_p is non-empty for all p <= current_size.
                # When we add x, new_size = current_size + 1.
                # We need to check primes p <= new_size.
                # For p <= current_size:
                #   If x % p is in missing_residues_p, then that residue is now covered.
                #   If missing_residues_p had size 1, and x covers it, then now missing_residues_p is empty -> INVALID.
                #   If missing_residues_p had size > 1, and x covers one, then size decreases -> still valid.
                # For p == new_size (if new_size is prime):
                #   We need to check if all residues are covered.
                
                # This logic is complex. Let's just do a full check but optimize by breaking early.
                # Since N=5000, and we might only add a few points, this might be okay.
                # But doing it for every candidate is expensive.
                
                # Alternative: Precompute missing residues for current point_set.
                # Let's do that outside the loop.
                pass
            
            # Let's implement the full check for correctness first, then optimize if needed.
            # Given time constraints, maybe just check the critical primes.
            
            # Actually, a simpler heuristic: 
            # Only try to add x if it doesn't immediately complete a class for small primes.
            
            # Let's just do the full admissibility check.
            if _is_admissible(new_set, primes):
                point_set = set(new_set)
                current_size = new_size
                candidates.remove(x)
                improved = True
                break # Restart loop with updated set? Or continue?
                # Restarting is safer to ensure we check against the new set.
                # But we can just continue if we update point_set.
                # However, the 'candidates' list is static.
                # Let's break and restart the while loop.
                
    return list(point_set)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_points = None
    best_size = -1
    
    # Precompute primes up to N
    # PRIMES is global
    
    # Run multiple randomized restarts with the greedy min-removal strategy
    # The key idea from inspirations is randomization and multiple restarts.
    # Also, the "conflict graph" idea suggests looking at local density.
    # Greedy min-removal is equivalent to choosing the least "conflict-heavy" class to remove.
    
    # We will try a range of seeds.
    # How many restarts can we do?
    # Each restart takes roughly O(N * P) time. N=5000, P~669.
    # np.bincount is fast. Let's estimate 1ms per restart?
    # If 1ms, we can do 200,000 restarts in 200s.
    # Let's try 1000 restarts first to see timing.
    
    max_restarts = 5000
    seed = 42
    
    for i in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        current_seed = seed + i
        points = _greedy_remove_min(None, PRIMES, current_seed, deadline)
        
        size = len(points)
        
        if size > best_size:
            best_size = size
            best_points = points.copy()
            
            # Try to polish the best result found so far?
            # Polishing is expensive, so only do it if we have time and significant improvement.
            # Or just do it at the end on the best result.
            
    # Final polish on the best result found
    if best_points is not None and time.time() < deadline - 5:
        # Sort points for consistency? Not needed for set operations.
        polished = _polish(best_points, PRIMES, deadline)
        if len(polished) > best_size:
            best_points = polished
            best_size = len(polished)
            
    if best_points is None:
        return [0]
        
    return best_points

# Quick sanity check
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")
    # Verify admissibility
    valid = _is_admissible(res, PRIMES)
    print(f"Valid: {valid}")
