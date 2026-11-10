import time
import math
import random
import numpy as np
from fractions import Fraction

_CAP = 100000
_DEADLINE = 110  # Safety margin

def _sieve_primes(n):
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(pairs):
    """
    pairs: list of [p, r]
    Returns largest L such that 1..L are covered.
    """
    if not pairs:
        return 0
    
    covered = np.zeros(_CAP + 1, dtype=bool)
    for p, r in pairs:
        # Mark r, r+p, r+2p...
        # Ensure r is in 0..p-1
        if r < 0:
            r %= p
        covered[r::p] = True
    
    # Find first index i >= 1 where covered[i] is False
    # covered[1:] corresponds to integers 1..CAP
    # If all True, return CAP
    # else return index of first False
    
    # np.where returns indices in the array passed
    # We pass covered[1:]
    # Index 0 in result corresponds to covered[1]
    # If covered[1] is False, index 0 is returned. L = 0.
    # If covered[1] is True, covered[2] is False, index 1 is returned. L = 1.
    # So L = index returned.
    
    uncovered = np.where(covered[1:] == False)[0]
    if len(uncovered) == 0:
        return _CAP
    return uncovered[0]

def _check_budget(pairs):
    """
    Check if sum of 1/p <= 1.
    Uses float for speed.
    """
    s = 0.0
    for p, r in pairs:
        s += 1.0 / p
    return s <= 1.0

def _random_residue(p):
    return random.randint(0, p - 1)

def _initial_solution():
    # Start with a set of small primes that fits budget
    # Greedy: add smallest primes first
    primes = _sieve_primes(1000)
    current_pairs = []
    current_sum = 0.0
    for p in primes:
        if current_sum + 1.0/p <= 1.0:
            current_pairs.append([p, _random_residue(p)])
            current_sum += 1.0/p
        else:
            # If we can't add this prime, maybe skip it and try larger ones?
            # For now, just stop or continue checking?
            # Greedy small primes is a good heuristic for initial structure
            pass
            
    # If we didn't get many, ensure we have at least a few
    if len(current_pairs) < 5:
         # Fallback to seed-like solution
         primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
         current_pairs = [[p, (i + 1) % p] for i, p in enumerate(primes)]
         
    return current_pairs

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    primes = _sieve_primes(1000)
    prime_set = set(primes)
    
    # Initial solution
    current_pairs = _initial_solution()
    current_L = _compute_L(current_pairs)
    
    best_pairs = [p.copy() for p in current_pairs]
    best_L = current_L
    
    # Local Search
    while time.time() < deadline:
        # Mutate
        candidate = [p.copy() for p in current_pairs]
        
        # Choose mutation type
        mut_type = random.randint(0, 3)
        
        if mut_type == 0 and candidate:
            # Change residue
            idx = random.randint(0, len(candidate) - 1)
            p, r = candidate[idx]
            new_r = _random_residue(p)
            candidate[idx][1] = new_r
        elif mut_type == 1 and candidate:
            # Change prime
            idx = random.randint(0, len(candidate) - 1)
            p, r = candidate[idx]
            # Pick a new prime not in candidate
            used_primes = {c[0] for c in candidate}
            available = [p2 for p2 in primes if p2 not in used_primes]
            if available:
                new_p = random.choice(available)
                # Keep same residue if possible, or random
                new_r = random.randint(0, new_p - 1)
                candidate[idx] = [new_p, new_r]
        elif mut_type == 2:
            # Add prime
            used_primes = {c[0] for c in candidate}
            available = [p2 for p2 in primes if p2 not in used_primes]
            if available:
                new_p = random.choice(available)
                # Check budget
                # Calculate current sum
                s = sum(1.0/c[0] for c in candidate)
                if s + 1.0/new_p <= 1.0:
                    new_r = _random_residue(new_p)
                    candidate.append([new_p, new_r])
        elif mut_type == 3 and len(candidate) > 1:
            # Remove prime
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
            
        # Validate budget
        if not _check_budget(candidate):
            continue
            
        # Evaluate
        cand_L = _compute_L(candidate)
        
        # Acceptance
        if cand_L > best_L:
            best_L = cand_L
            best_pairs = [p.copy() for p in candidate]
            current_pairs = candidate
        elif cand_L >= current_L - 5:
            # Allow some worsening to escape local minima
            current_pairs = candidate
        # else: reject
        
    # Final check and return
    # Ensure best_pairs is valid
    if not _check_budget(best_pairs):
         # Fallback to seed
         primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
         return [[p, (i + 1) % p] for i, p in enumerate(primes)]
         
    return best_pairs
