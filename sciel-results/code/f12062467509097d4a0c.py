import numpy as np
import time
import random
from collections import defaultdict

N = 5000
RANGE = N + 1

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _compute_admissible_size(residue_choices, max_prime_idx):
    """
    Given residue_choices[i] = residue to block for prime PRIMES[i] (i in range(max_prime_idx)),
    compute how many x in [0, N] are NOT blocked by any of these constraints.
    An x is blocked if for some i, x % PRIMES[i] == residue_choices[i].
    We want to maximize the number of unblocked x.
    """
    if max_prime_idx == 0:
        return RANGE
    
    # For each x, check if it's covered by any blocked residue class
    # Use inclusion-exclusion is too complex; just mark coverage
    covered = np.zeros(RANGE, dtype=bool)
    for i in range(max_prime_idx):
        p = PRIMES[i]
        r = residue_choices[i]
        # Mark all x in [0, RANGE) where x % p == r
        covered[r::p] = True
    
    return np.sum(~covered)

def _compute_admissible_set(residue_choices, max_prime_idx):
    """Return the list of admissible points given residue choices."""
    if max_prime_idx == 0:
        return list(range(RANGE))
    
    covered = np.zeros(RANGE, dtype=bool)
    for i in range(max_prime_idx):
        p = PRIMES[i]
        r = residue_choices[i]
        covered[r::p] = True
    
    return np.where(~covered)[0].tolist()

def _max_prime_for_size(sz):
    """Return the index (exclusive) of primes to consider for a set of size sz."""
    # We need primes p <= sz
    count = 0
    for p in PRIMES:
        if p <= sz:
            count += 1
        else:
            break
    return count

def _sa_on_residue_space(seed, deadline):
    """
    Simulated annealing on the space of residue choices.
    State: residue_choices array + max_prime_idx.
    Objective: maximize admissible size (number of unblocked points).
    Constraint: for the resulting set of size S, we need max_prime_idx = _max_prime_for_size(S).
    This creates a self-consistency requirement.
    """
    random.seed(seed)
    
    # Start with a greedy initialization
    # Greedy: for each prime in order, pick the residue that blocks the fewest currently-unblocked points
    residue_choices = np.zeros(len(PRIMES), dtype=int)
    max_pi = 0
    unblocked = np.ones(RANGE, dtype=bool)
    
    for i in range(len(PRIMES)):
        p = PRIMES[i]
        if p > np.sum(unblocked):
            break
        max_pi = i + 1
        # Count how many unblocked points are in each residue class
        counts = np.zeros(p, dtype=int)
        for r in range(p):
            mask = unblocked & (np.arange(RANGE) % p == r)
            counts[r] = np.sum(mask)
        # Pick residue with minimum count (blocks fewest)
        r = int(np.argmin(counts))
        residue_choices[i] = r
        # Update unblocked
        unblocked[r::p] = False
    
    current_size = np.sum(unblocked)
    current_max_pi = max_pi
    best_size = current_size
    best_choices = residue_choices.copy()
    best_max_pi = current_max_pi
    
    T = 5.0
    cooling = 0.9995
    steps = 0
    
    x_indices = np.arange(RANGE)
    
    while time.time() < deadline - 2 and steps < 30000:
        steps += 1
        
        # Generate a move: change one residue choice
        if current_max_pi == 0:
            continue
            
        i = random.randint(0, current_max_pi - 1)
        p = PRIMES[i]
        old_r = residue_choices[i]
        
        # Try a different residue
        if p == 1:
            continue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        # Compute new size efficiently
        # We need to recompute which points are unblocked
        # This is O(current_max_pi * RANGE/p) which is manageable
        covered = np.zeros(RANGE, dtype=bool)
        for j in range(current_max_pi):
            pj = PRIMES[j]
            rj = residue_choices[j] if j != i else new_r
            covered[rj::pj] = True
        
        new_size = np.sum(~covered)
        
        # Check self-consistency: max_prime_idx should match new_size
        required_max_pi = _max_prime_for_size(new_size)
        
        # If the required max_pi differs, we need to adjust
        # For now, accept only if self-consistent, or handle the transition
        if required_max_pi == current_max_pi:
            delta = new_size - current_size
            if delta > 0 or random.random() < np.exp(-delta / T):
                residue_choices[i] = new_r
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_choices = residue_choices.copy()
                    best_max_pi = current_max_pi
        elif required_max_pi > current_max_pi:
            # Size grew, need more primes to be considered
            # Extend residue_choices with greedy choices for new primes
            old_max_pi = current_max_pi
            current_max_pi = required_max_pi
            # Recompute covered with new primes using greedy init for new ones
            unblocked_new = np.ones(RANGE, dtype=bool)
            for j in range(current_max_pi):
                pj = PRIMES[j]
                if j < old_max_pi:
                    rj = residue_choices[j]
                else:
                    # Greedy: pick residue that blocks fewest currently unblocked
                    counts = np.zeros(pj, dtype=int)
                    for r in range(pj):
                        mask = unblocked_new & (x_indices % pj == r)
                        counts[r] = np.sum(mask)
                    rj = int(np.argmin(counts))
                    residue_choices[j] = rj
                unblocked_new[rj::pj] = False
            
            new_size2 = np.sum(unblocked_new)
            # Re-check self-consistency
            required2 = _max_prime_for_size(new_size2)
            if required2 == current_max_pi:
                current_size = new_size2
                if current_size > best_size:
                    best_size = current_size
                    best_choices = residue_choices.copy()
                    best_max_pi = current_max_pi
            else:
                # Inconsistent, revert
                residue_choices[i] = old_r
                current_max_pi = old_max_pi
        else:
            # Size shrank, fewer primes needed
            # Truncate residue_choices
            old_max_pi = current_max_pi
            current_max_pi = required_max_pi
            # Recompute with truncated
            unblocked_new = np.ones(RANGE, dtype=bool)
            for j in range(current_max_pi):
                pj = PRIMES[j]
                rj = residue_choices[j]
                unblocked_new[rj::pj] = False
            new_size2 = np.sum(unblocked_new)
            required2 = _max_prime_for_size(new_size2)
            if required2 == current_max_pi:
                current_size = new_size2
                if current_size > best_size:
                    best_size = current_size
                    best_choices = residue_choices.copy()
                    best_max_pi = current_max_pi
            else:
                # Revert
                residue_choices[i] = old_r
                current_max_pi = old_max_pi
            # Note: we don't track best here if size decreased, but current state is valid
        
        T *= cooling
        if T < 0.01:
            break
    
    return _compute_admissible_set(best_choices, best_max_pi), best_size

def _greedy_construct(seed, deadline):
    """
    Construct an admissible tuple using greedy residue removal.
    Returns the list of points.
    """
    random.seed(seed)
    active = np.ones(RANGE, dtype=bool)
    x_indices = np.arange(RANGE)
    current_size = RANGE
    
    for p in PRIMES:
        if time.time() > deadline:
            break
        if p > current_size:
            break
        
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        residues = active_indices % p
        if len(residues) == 0:
            break
        
        counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(counts)
        
        if min_count == 0:
            continue
        
        candidate_classes = np.where(counts == min_count)[0]
        
        r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
        
        mask = (x_indices % p == r) & active
        current_size -= np.sum(mask)
        active = active & ~mask
        
        if current_size == 0:
            break
    
    return np.where(active)[0].tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_size = 0
    best_points = []
    
    # First, do a few greedy constructions to get a baseline
    for seed in range(1, 11):
        if time.time() > deadline - 30:
            break
        points = _greedy_construct(seed, deadline)
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
    
    # Now do SA on residue space with multiple restarts
    sa_restarts = 0
    while time.time() < deadline - 5:
        sa_restarts += 1
        sa_deadline = time.time() + min(25, deadline - time.time() - 3)
        if sa_deadline <= time.time():
            break
        points, size = _sa_on_residue_space(sa_restarts * 1000 + 42, sa_deadline)
        if size > best_size:
            best_size = size
            best_points = points
    
    return best_points
