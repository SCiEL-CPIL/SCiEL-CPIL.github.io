"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(2, limit + 1) if sieve[i]]


def _residue_counts(arr, p):
    """Return list of length p with counts of each residue class mod p."""
    rem = arr % p
    return np.bincount(rem, minlength=p)


def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple: for every prime p in primes_to_check,
    some residue class mod p is missed."""
    if len(points) == 0:
        return False
    for p in primes_to_check:
        if p > len(points):
            break
        counts = _residue_counts(np.asarray(points), p)
        if 0 in counts:
            continue
        # all p classes covered -> not admissible for this p
        return False
    return True


def construct():
    start_time = time.time()
    primes_all = _primes(N)
    
    # We need primes p <= len(tuple). Since we're in [0,5000], max tuple size is 5001.
    # Primes up to 5001 are relevant.
    primes_check = [p for p in primes_all if p <= 5001]
    
    # Strategy: Start with a large candidate set, then greedily remove elements
    # to make it admissible. The key insight is that for large primes p > sqrt(N),
    # if we have fewer than p elements, we can't cover all residue classes, so those
    # primes are automatically satisfied. The binding constraints are small primes.
    
    # Initial candidate: all integers in [0, N]
    points = list(range(N + 1))
    
    # For admissibility, we need: for every prime p <= len(points), some residue mod p is missed.
    # This is equivalent to: the tuple does NOT cover all residue classes mod p for any p <= len(tuple).
    
    # A better approach: work backwards. Start with all elements, and for each prime p
    # (in increasing order), if all p residue classes are present, remove the least-populated
    # class. But we want to keep as many elements as possible, so we remove the smallest
    # class.
    
    # However, removing elements for one prime may affect others. We need an iterative
    # approach.
    
    # Let's use a smarter initialization: 
    # For the constraint to be tight, we want to miss one residue class for each small prime.
    # A classic construction: pick a set that is a union of arithmetic progressions 
    # that miss specific residues.
    
    # Actually, let's think about this differently. The maximum admissible tuple in [0, N]
    # is related to the Jacobsthal function. But for now, let's use a greedy removal approach.
    
    # Better idea: For each prime p <= 5001, we must miss at least one residue class.
    # To maximize size, we want to miss the residue class with the fewest elements.
    # But the choice of which residue to miss for one prime interacts with others.
    
    # Let's try: start with all elements. For primes in increasing order, if all residue
    # classes mod p are covered, remove the least-populated residue class. Repeat until
    # admissible.
    
    points_set = set(range(N + 1))
    
    # Iterate until admissible
    max_iter = 100
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        current_len = len(points_set)
        if current_len == 0:
            break
        primes_needed = [p for p in primes_check if p <= current_len]
        if not primes_needed:
            break
        
        violated = False
        for p in primes_needed:
            # Count residues of points_set mod p
            arr = np.fromiter(points_set, dtype=np.int64)
            counts = _residue_counts(arr, p)
            if 0 in counts:
                continue
            # All classes covered; find least populated class
            min_count = counts.min()
            # Find all classes with min count; pick one (use smallest index for determinism)
            classes_to_remove = np.where(counts == min_count)[0]
            # Pick the first one
            target_class = classes_to_remove[0]
            # Remove all elements in points_set with v % p == target_class
            to_remove = [v for v in points_set if v % p == target_class]
            points_set.difference_update(to_remove)
            violated = True
            break
        
        if not violated:
            break
    
    result = list(points_set)
    
    # Verify admissibility
    if not _is_admissible(result, primes_check):
        # Fallback: use the original seed algorithm result
        points_orig = [1] + [p for p in primes_all if p * p > N]
        changed = True
        while changed:
            changed = False
            for p in _primes(len(points_orig)):
                classes = {}
                for v in points_orig:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    points_orig = [v for v in points_orig if v not in smallest]
                    changed = True
                    break
        result = points_orig
    
    return result

# EVOLVE-BLOCK-END
